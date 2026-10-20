"""
siteadapt/evaluate.py
----------------------
Entrenamiento, evaluación y selección de modelos de site-adaptation, y
reconstrucción y re-verificación de la serie terrestre corregida.

Todos los modelos se entrenan en TRAIN y se evalúan ÚNICAMENTE en TEST:
ninguna métrica de desempeño se calcula sobre datos que el modelo vio
durante el entrenamiento (evita la fuga de datos y la sobreestimación de
la mejora).

Selección: se descarta todo modelo cuyas predicciones sobre el periodo
de aplicación completo violen las pruebas físicas; entre los válidos gana
el de menor RMSE en test. Con el modelo ganador se construyen las
variantes total y selectiva de la serie corregida, que se re-verifican
con el mismo QC aplicado a la estación terrestre.
"""

import numpy as np
import pandas as pd

from . import models as mdl
from .. import metrics as met
from ..qc import physical as qcf
from ..qc import expert as qce
from ..qc import usability as qcu


def entrenar_y_evaluar(train: pd.DataFrame, test: pd.DataFrame, target: str,
                        columnas_predictoras: list[str], random_state: int = 42):
    """
    Entrena TODOS los modelos candidatos sobre `train` y los evalúa sobre `test`.

    Devuelve:
      - tabla_metricas: DataFrame con MAE/RMSE/MBE/rMBE/rRMSE/Pearson/R2/n
        por modelo, calculadas SOLO sobre test y ordenadas por RMSE (mejor primero).
      - predicciones_test: DataFrame con datetime, el valor real y la
        predicción de cada modelo sobre test (para graficar después).
      - modelos_entrenados: dict {nombre: modelo} ya entrenado, para
        aplicarlo o inspeccionarlo (p. ej. importancia de variables).
    """
    X_train, y_train = train, train[target]
    X_test, y_test = test, test[target]

    modelos = mdl.construir_modelos(columnas_predictoras, random_state=random_state)

    filas_metricas = []
    predicciones_test = pd.DataFrame({
        "datetime": test["datetime"].values,
        target: y_test.values,
    })

    modelos_entrenados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        predicciones_test[nombre] = y_pred
        modelos_entrenados[nombre] = modelo

        fila = {"modelo": nombre}
        fila.update(met.metricas_regresion(y_test, y_pred))
        filas_metricas.append(fila)

    tabla_metricas = pd.DataFrame(filas_metricas)
    tabla_metricas = tabla_metricas.sort_values(["RMSE"], ascending=True).reset_index(drop=True)
    tabla_metricas.insert(0, "ranking", range(1, len(tabla_metricas) + 1))

    return tabla_metricas, predicciones_test, modelos_entrenados


def tabla_mejora_vs_baseline(tabla_metricas: pd.DataFrame,
                              nombre_baseline: str = "Baseline (Solcast sin corregir)") -> pd.DataFrame:
    """
    Para cada modelo, calcula el % de mejora respecto al baseline (fuente
    satelital sin corregir) en RMSE, MAE y R2, a partir de las métricas de
    `tabla_metricas` (que deben provenir SOLO de test).
    """
    fila_base = tabla_metricas.loc[tabla_metricas["modelo"] == nombre_baseline].iloc[0]

    filas = []
    for _, fila in tabla_metricas.iterrows():
        filas.append({
            "modelo": fila["modelo"],
            "RMSE": fila["RMSE"],
            "mejora_RMSE_%": met.porcentaje_mejora(fila_base["RMSE"], fila["RMSE"]),
            "MAE": fila["MAE"],
            "mejora_MAE_%": met.porcentaje_mejora(fila_base["MAE"], fila["MAE"]),
            "R2": fila["R2"],
            "mejora_R2_%": met.porcentaje_mejora(fila_base["R2"], fila["R2"], mas_alto_es_mejor=True),
        })
    return pd.DataFrame(filas)


# ================================================================
#  VALIDEZ FÍSICA, SELECCIÓN Y RECONSTRUCCIÓN. Funciones reutilizables
#  sobre cualquier estación terrestre objetivo.
# ================================================================

# No se excluye ningún modelo a priori por su familia: el filtro de
# validez física sobre TODO el periodo de aplicación
# (evaluar_consistencia_fisica_aplicacion) comprueba empíricamente si un
# modelo produce valores imposibles, y es el único criterio de exclusión.


def evaluar_consistencia_fisica(predicciones_test: pd.DataFrame, target: str,
                                 minimo_fisico: float) -> pd.DataFrame:
    """
    Criterio simple de validez física: para cada modelo en
    `predicciones_test` (columnas distintas de datetime/target), cuenta
    cuántas predicciones de TEST caen por debajo del mínimo físico BSRN
    (config.yaml: qc_fisico.minimo_irradiancia). Un modelo es válido si no
    tiene ninguna.
    """
    nombres_modelos = [c for c in predicciones_test.columns if c not in ("datetime", target)]
    total = len(predicciones_test)
    filas = []
    for nombre_m in nombres_modelos:
        n_bajo_minimo = int((predicciones_test[nombre_m] < minimo_fisico).sum())
        filas.append({
            "modelo": nombre_m,
            "n_bajo_minimo_fisico": n_bajo_minimo,
            "pct_bajo_minimo_fisico": round(100 * n_bajo_minimo / total, 3) if total else float("nan"),
            "fisicamente_valido": n_bajo_minimo == 0,
        })
    return pd.DataFrame(filas)


def predecir_aplicacion_todos_modelos(aplicacion: pd.DataFrame, modelos_entrenados: dict) -> pd.DataFrame:
    """
    Aplica CADA modelo ya entrenado (no solo el ganador) sobre TODO el
    periodo de aplicación (no solo el 30 % de prueba), para verificar la
    validez física en el uso real antes de elegir el modelo ganador (ver
    `evaluar_consistencia_fisica_aplicacion()`).

    Devuelve un DataFrame con "datetime" + una columna por modelo.
    """
    out = pd.DataFrame({"datetime": aplicacion["datetime"].values})
    for nombre, modelo in modelos_entrenados.items():
        out[nombre] = modelo.predict(aplicacion)
    return out


def evaluar_consistencia_fisica_aplicacion(predicciones_aplicacion: pd.DataFrame,
                                            aplicacion_con_geometria: pd.DataFrame,
                                            minimo_fisico: float,
                                            tolerancia_nocturna_pct: float = 0.01) -> pd.DataFrame:
    """
    Criterio riguroso de validez física: en vez de revisar solo el mínimo
    físico sobre el 30 % de prueba, aplica a la predicción de CADA modelo,
    sobre TODO el periodo de aplicación, las MISMAS pruebas físicas BSRN
    que se usan para una estación terrestre
    (`qc.physical.aplicar_qc_fisico(..., "terrestre")`: negativo, límite
    físicamente posible PPL, límite extremadamente raro ERL y nocturno).

    Por qué existe: un modelo puede no tener ninguna predicción bajo el
    mínimo en el 30 % de prueba y aun así producir un porcentaje relevante
    de valores físicamente imposibles (PPL/ERL, no solo negativos) al
    aplicarse sobre el año completo, donde extrapola fuera del rango de
    condiciones vistas en prueba.

    TOLERANCIA (solo para la prueba nocturna): neg_GHI/ppl_GHI/erl_GHI
    (valores físicamente imposibles) se exigen SIEMPRE en cero. La prueba
    nocturna (`flag_noche_GHI`) admite hasta `tolerancia_nocturna_pct` %
    del periodo de aplicación (0.01 % por defecto, ~10 de ~105 000
    muestras de 5 min en un año). Las violaciones nocturnas aisladas
    suelen ser crepusculares (sol apenas bajo el horizonte y predicciones
    ligeramente por encima de `max_irradiancia_noche`): luz residual del
    crepúsculo, un fenómeno físico real y no una extrapolación. Sin esta
    tolerancia, un modelo por lo demás válido quedaría descalificado por
    un puñado de casos triviales.

    `aplicacion_con_geometria` debe traer 'zenith'/'elevation'/'dni_extra'
    SIN sufijo (calculados con `solar.agregar_geometria_solar()` sobre su
    propio datetime), en el mismo orden de filas que
    `predicciones_aplicacion`. Devuelve una tabla por modelo con conteos,
    porcentajes de invalidez y la columna booleana `fisicamente_valido`.
    """
    cols_geo = [c for c in ("zenith", "elevation", "dni_extra") if c in aplicacion_con_geometria.columns]
    total = len(predicciones_aplicacion)
    max_noche_tolerado = max(0, round(total * tolerancia_nocturna_pct / 100))
    filas = []
    for nombre_m in [c for c in predicciones_aplicacion.columns if c != "datetime"]:
        tmp = aplicacion_con_geometria[cols_geo].copy()
        tmp["GHI"] = predicciones_aplicacion[nombre_m].to_numpy()
        tmp = qcf.aplicar_qc_fisico(tmp, "terrestre")
        cols_flag_estrictas = [c for c in tmp.columns
                                if c.startswith("flag_") and c != "flag_noche_GHI"]
        es_invalido_estricto = (tmp[cols_flag_estrictas] == "FLAG").any(axis=1)
        n_estricto = int(es_invalido_estricto.sum())
        n_noche = int((tmp.get("flag_noche_GHI", "OK") == "FLAG").sum()) if "flag_noche_GHI" in tmp.columns else 0
        n_invalido = n_estricto + n_noche
        es_valido = (n_estricto == 0) and (n_noche <= max_noche_tolerado)
        n_bajo_minimo = int((tmp["GHI"] < minimo_fisico).sum())
        filas.append({
            "modelo": nombre_m,
            "n_bajo_minimo_fisico": n_bajo_minimo,
            "pct_bajo_minimo_fisico": round(100 * n_bajo_minimo / total, 3) if total else float("nan"),
            "n_fisicamente_invalido": n_invalido,
            "pct_fisicamente_invalido": round(100 * n_invalido / total, 3) if total else float("nan"),
            "n_noche_tolerado": max_noche_tolerado,
            "fisicamente_valido": es_valido,
        })
    return pd.DataFrame(filas)


def elegir_mejor_modelo(tabla_metricas: pd.DataFrame, predicciones_test: pd.DataFrame,
                         target: str, minimo_fisico: float,
                         predicciones_aplicacion: pd.DataFrame | None = None,
                         aplicacion_con_geometria: pd.DataFrame | None = None) -> tuple[str, str, pd.DataFrame]:
    """
    Elige el modelo ganador: se descarta cualquier modelo que produzca
    valores físicamente inválidos; entre los que quedan, gana el de menor
    RMSE en test. No hay exclusión por familia de modelo. Si ningún modelo
    pasa el filtro, se usa como último recurso el de menor RMSE puro (la
    columna `fisicamente_valido` de la tabla devuelta permite detectarlo).

    Si se pasan `predicciones_aplicacion` (ver
    `predecir_aplicacion_todos_modelos()`) Y `aplicacion_con_geometria`,
    la validez física se evalúa con el criterio RIGUROSO sobre TODO el
    periodo de aplicación (`evaluar_consistencia_fisica_aplicacion()`) en
    lugar del criterio simple sobre el 30 % de prueba. Es la opción
    recomendada, porque el criterio simple puede dejar pasar un modelo que
    extrapola mal fuera del rango de prueba.

    Devuelve (nombre_ganador, nombre_mejor_rmse_puro, tabla_metricas_enriquecida).
    """
    if predicciones_aplicacion is not None and aplicacion_con_geometria is not None:
        tabla_validez = evaluar_consistencia_fisica_aplicacion(
            predicciones_aplicacion, aplicacion_con_geometria, minimo_fisico)
    else:
        tabla_validez = evaluar_consistencia_fisica(predicciones_test, target, minimo_fisico)

    tabla = tabla_metricas.merge(tabla_validez, on="modelo", how="left")

    mejor_rmse_puro = tabla.sort_values("RMSE").iloc[0]["modelo"]
    candidatos = tabla[tabla["fisicamente_valido"]].sort_values("RMSE")

    ganador = candidatos.iloc[0]["modelo"] if not candidatos.empty else mejor_rmse_puro
    return ganador, mejor_rmse_puro, tabla


def construir_serie_corregida(aplicacion: pd.DataFrame, modelo, nombre_objetivo: str) -> pd.DataFrame:
    """
    Aplica `modelo` (ya entrenado en el conjunto supervisado) sobre TODO
    `aplicacion` (todos los instantes con dato satelital válido, tenga o no
    dato la estación en ese momento) y construye las 2 variantes de la
    serie corregida:

      GHI_corregido_total     -- la predicción del modelo en TODOS los
                                  instantes, sin excepción.
      GHI_corregido_selectivo -- conserva el dato medido cuando
                                  origen_dato=="medido_ok"; en el resto
                                  (medido_flag / sin_dato) usa la
                                  predicción del modelo.

    Devuelve una copia de `aplicacion` con esas 2 columnas nuevas.
    """
    out = aplicacion.copy()
    pred = modelo.predict(out)
    out["GHI_corregido_total"] = pred

    col_medido = f"GHI_{nombre_objetivo}"
    medido = out[col_medido] if col_medido in out.columns else pd.Series(np.nan, index=out.index)
    es_ok = out["origen_dato"] == "medido_ok"
    out["GHI_corregido_selectivo"] = np.where(es_ok, medido.to_numpy(), pred)

    return out


def aplicar_qc_a_serie(df: pd.DataFrame, col_ghi: str, prefijo: str) -> tuple[pd.DataFrame, list[str], str]:
    """
    Aplica qc.physical + qc.expert (modo estación terrestre, una sola
    componente GHI) a la columna `col_ghi` de `df` y calcula la usabilidad.
    Se usa para re-verificar la serie cruda y las series corregidas de la
    fuente terrestre con el mismo QC.

    IMPORTANTE: `df` debe tener las columnas de geometría solar SIN
    sufijo (zenith, elevation, dni_extra, ghi_clear, ...) calculadas con
    `solar.agregar_geometria_solar()` sobre su propia columna datetime, no
    las columnas con sufijo de la fuente satelital (esas son solo
    predictoras).

    Devuelve (df_con_flags, columnas_flag_nuevas, columna_usabilidad_nueva).
    Las columnas quedan prefijadas con `{prefijo}_` para poder evaluar
    varias series (crudo, corregido_total, corregido_selectivo) en el
    mismo DataFrame sin que se sobrescriban entre sí.
    """
    df = df.copy()
    df_tmp = df.rename(columns={col_ghi: "GHI"})
    df_tmp = qcf.aplicar_qc_fisico(df_tmp, "terrestre")   # genérico: una sola componente GHI
    df_tmp = qce.aplicar_qc_experto(df_tmp, "terrestre")

    columnas_flag_orig = [c for c in df_tmp.columns if c.startswith("flag_")]
    cols_propias = []
    for c in columnas_flag_orig:
        nueva_col = f"{prefijo}_{c}"
        df[nueva_col] = df_tmp[c].to_numpy()
        cols_propias.append(nueva_col)

    col_usab = f"usabilidad_{prefijo}"
    df[col_usab] = qcu.calcular_usabilidad(df, columnas_flag=cols_propias).to_numpy()
    return df, cols_propias, col_usab


def resumen_flags_serie(df: pd.DataFrame, cols_propias: list[str], prefijo: str) -> pd.DataFrame:
    """Cuenta OK/FLAG/NO_EVALUABLE por prueba para una serie ya evaluada con `aplicar_qc_a_serie()`."""
    filas = []
    total = len(df)
    for c in cols_propias:
        conteo = df[c].value_counts()
        n_flag = int(conteo.get("FLAG", 0))
        filas.append({
            "serie": prefijo,
            "prueba": c.replace(f"{prefijo}_flag_", ""),
            "OK": int(conteo.get("OK", 0)),
            "FLAG": n_flag,
            "NO_EVALUABLE": int(conteo.get("NO_EVALUABLE", 0)),
            "pct_flag": round(100 * n_flag / total, 3) if total else float("nan"),
        })
    return pd.DataFrame(filas)
