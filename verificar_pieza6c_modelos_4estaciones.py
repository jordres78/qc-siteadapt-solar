"""
verificar_pieza6c_modelos_4estaciones.py
----------------------------------------
Paso 6 del pipeline: entrenamiento, evaluacion y seleccion del modelo de
adaptacion al sitio que estima el GHI de la estacion terrestre a partir
de la fuente satelital. Los nombres de los scripts conservan una
numeracion por paso del pipeline.

Para cada estacion de site_adaptation.reconstruccion.fuentes_terrestres:
  1. Ejecuta los pasos 1-4 y el emparejamiento a la resolucion de
     config.yaml (5 min por defecto), con division temporal 70/30.
  2. Entrena los nueve modelos sobre train y calcula metricas solo sobre
     test.
  3. Selecciona el modelo: se descarta todo modelo cuyas predicciones
     sobre el periodo completo de aplicacion violen las pruebas fisicas
     (tolerancia nocturna de 0,01 %); entre los validos gana el de menor
     RMSE en test, sin exclusion por familia de modelo.

La reconstruccion de la serie y la reaplicacion del QC se realizan en
verificar_pieza7b_reconstruccion_4estaciones.py, que usa el mismo
criterio de seleccion y por tanto el mismo modelo.

Entradas: config.yaml y los archivos crudos de data/raw/.
Salidas:
  - outputs/tablas/reconstruccion_metricas_modelos.csv.
  - outputs/tablas/reconstruccion_<estacion>_mejora_vs_baseline.csv.
  - outputs/tablas/reconstruccion_<estacion>_importancia_variables.csv
    (vacia si el modelo seleccionado no tiene importancia de variables).
  - outputs/tablas/reconstruccion_<estacion>_coeficientes.csv (solo si el
    modelo seleccionado es lineal).
  - data/processed/reconstruccion_<estacion>_predicciones_test.csv.

Como ejecutarlo (con el entorno virtual activo):
    python verificar_pieza6c_modelos_4estaciones.py
"""

import pandas as pd

from qcsolar.io import cargar_solcast, CARGADORES_TERRESTRES
from qcsolar.solar import agregar_geometria_solar
from qcsolar.qc import physical as qcf
from qcsolar.qc import expert as qce
from qcsolar.qc import usability as qcu
from qcsolar.siteadapt import prepare as prep
from qcsolar.siteadapt import evaluate as ev
from qcsolar.config import CONFIG
from qcsolar import report

pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.3f}".format)

RECON_CFG = CONFIG["site_adaptation"]["reconstruccion"]
FUENTES_TERRESTRES = RECON_CFG["fuentes_terrestres"]
EXCLUSIONES = RECON_CFG.get("exclusiones", {})
MINIMO_FISICO = CONFIG["qc_fisico"]["minimo_irradiancia"]
FREQ = RECON_CFG.get("resolucion", "5min")

COLUMNAS_PREDICTORAS = [
    "GHI_solcast", "DNI_solcast", "DHI_solcast", "GTI_solcast",
    "TEMP_solcast", "RH_solcast", "WS_solcast", "CLOUD_OPACITY_solcast",
    "zenith_solcast", "elevation_solcast", "hour", "dayofyear", "month",
]


def pipeline_completo(nombre, df):
    """Ejecuta los pasos 1-4 del pipeline (geometria solar, QC fisico, QC
    experto, usabilidad) sobre una fuente cruda, en el orden requerido."""
    print(f"   [{nombre}] geometria solar...")
    df = agregar_geometria_solar(df)
    print(f"   [{nombre}] QC fisico...")
    df = qcf.aplicar_qc_fisico(df, nombre)
    print(f"   [{nombre}] QC experto...")
    df = qce.aplicar_qc_experto(df, nombre)
    print(f"   [{nombre}] usabilidad...")
    df["usabilidad"] = qcu.calcular_usabilidad(df)
    return df


def _tabla_importancia(modelo, columnas):
    """Devuelve la tabla de importancia de variables de un modelo de arboles
    ya entrenado, o None si el modelo no expone feature_importances_ (p. ej.,
    un modelo lineal)."""
    if not hasattr(modelo, "feature_importances_") or modelo.feature_importances_ is None:
        return None
    return pd.DataFrame({
        "variable": columnas,
        "importancia": modelo.feature_importances_,
    }).sort_values("importancia", ascending=False).reset_index(drop=True)


def main():
    """Entrena y evalua los modelos para cada estacion, selecciona el modelo
    y guarda las tablas de metricas, mejora, importancia y predicciones."""
    print("=" * 70)
    print(" PIEZA 6c - Modelos de reconstruccion fuente satelital -> terrestre(s)")
    print("=" * 70)

    # Cada estacion terrestre tiene su propia serie satelital asociada
    # (config.yaml: satelital_por_fuente). La serie satelital procesada se
    # guarda en cache para no reprocesarla si varias estaciones la comparten.
    _cache_solcast = {}

    def _solcast_para(clave_terrestre):
        """Devuelve la serie satelital asociada a la estacion, ya procesada
        con los pasos 1-4 (se calcula una sola vez por serie satelital)."""
        clave_sat = CONFIG.get("satelital_por_fuente", {}).get(clave_terrestre, "solcast_raw")
        if clave_sat not in _cache_solcast:
            print(f"\n Procesando fuente SATELITAL ({clave_sat}, pareja de '{clave_terrestre}')...")
            _cache_solcast[clave_sat] = pipeline_completo(clave_sat, cargar_solcast(clave_terrestre))
        return _cache_solcast[clave_sat]

    todas_metricas = []
    for clave in FUENTES_TERRESTRES:
        if clave not in CARGADORES_TERRESTRES:
            continue
        print(f"\n{'=' * 70}\n ESTACION: {clave.upper()}\n{'=' * 70}")
        sol = _solcast_para(clave)
        est = pipeline_completo(clave, CARGADORES_TERRESTRES[clave]())

        excl = EXCLUSIONES.get(clave, {}) or {}
        supervisado, aplicacion = prep.emparejar_horario_generico(
            sol, est, nombre_objetivo=clave,
            excluir_desde=excl.get("desde"), excluir_hasta=excl.get("hasta"),
            freq=FREQ,
        )
        if supervisado.empty:
            print(f"   AVISO: 0 muestras supervisadas para {clave}. Se omite esta estacion.")
            continue

        train, test = prep.split_temporal(supervisado)
        print(f"   Muestras supervisadas ({FREQ}): {len(supervisado):,} (train={len(train):,}, test={len(test):,})")

        target = f"GHI_{clave}"
        columnas_disp = [c for c in COLUMNAS_PREDICTORAS if c in supervisado.columns]

        tabla_metricas, predicciones_test, modelos_entrenados = ev.entrenar_y_evaluar(
            train, test, target=target, columnas_predictoras=columnas_disp)
        print("\n   --- Ranking de modelos (test) ---")
        print(tabla_metricas.to_string(index=False))

        # Filtro de validez fisica sobre el periodo completo de aplicacion
        # (no solo sobre test): un modelo sin violaciones en test puede
        # producirlas en el resto del periodo. Es el mismo criterio que usa
        # la etapa de reconstruccion, por lo que ambas eligen el mismo modelo.
        aplicacion_geo = agregar_geometria_solar(aplicacion.reset_index(drop=True), "datetime")
        predicciones_aplicacion_todos = ev.predecir_aplicacion_todos_modelos(aplicacion, modelos_entrenados)
        ganador, mejor_rmse_puro, tabla_metricas = ev.elegir_mejor_modelo(
            tabla_metricas, predicciones_test, target, MINIMO_FISICO,
            predicciones_aplicacion=predicciones_aplicacion_todos,
            aplicacion_con_geometria=aplicacion_geo,
        )
        print(f"\n   Mejor modelo por RMSE puro: {mejor_rmse_puro}")
        print(f"   Modelo elegido (menor RMSE entre los fisicamente validos sobre TODO el periodo): {ganador}")

        tabla_metricas.insert(0, "estacion", clave)
        tabla_metricas["modelo_ganador"] = tabla_metricas["modelo"] == ganador
        todas_metricas.append(tabla_metricas)

        tabla_mejora = ev.tabla_mejora_vs_baseline(tabla_metricas.drop(columns=["estacion", "modelo_ganador"]))
        tabla_mejora.insert(0, "estacion", clave)
        report.guardar_tabla(tabla_mejora, f"reconstruccion_{clave}_mejora_vs_baseline.csv")

        importancia = _tabla_importancia(modelos_entrenados[ganador], columnas_disp)
        if importancia is not None:
            importancia.insert(0, "estacion", clave)
            print(f"\n   --- Importancia de variables ({ganador}) ---")
            print(importancia.to_string(index=False))
            report.guardar_tabla(importancia, f"reconstruccion_{clave}_importancia_variables.csv")
        else:
            # El modelo seleccionado puede ser lineal (sin importancia por
            # impureza). Se guarda la tabla de importancia vacia, para que no
            # quede una tabla de una ejecucion previa, y se guardan los
            # coeficientes de la recta, que son su equivalente interpretable.
            report.guardar_tabla(pd.DataFrame(columns=["estacion", "variable", "importancia"]),
                                 f"reconstruccion_{clave}_importancia_variables.csv")
            _lin = getattr(modelos_entrenados[ganador], "modelo", None)
            if _lin is not None and hasattr(_lin, "coef_"):
                _vars = list(getattr(_lin, "feature_names_in_", [f"x{i}" for i in range(len(_lin.coef_))]))
                coefs = pd.DataFrame({"estacion": clave, "modelo": ganador,
                                      "termino": ["intercepto"] + _vars,
                                      "valor": [float(_lin.intercept_)] + [float(c) for c in _lin.coef_]})
                print(f"\n   --- Coeficientes ({ganador}) ---")
                print(coefs.to_string(index=False))
                report.guardar_tabla(coefs, f"reconstruccion_{clave}_coeficientes.csv")

        report.guardar_dataset(predicciones_test, f"reconstruccion_{clave}_predicciones_test.csv",
                                etapa="processed")

    if todas_metricas:
        report.guardar_tabla(pd.concat(todas_metricas, ignore_index=True),
                              "reconstruccion_metricas_modelos.csv")

    print("\n" + "=" * 70)
    print(" PIEZA 6c LISTA. Revisa outputs/tablas/reconstruccion_metricas_modelos.csv")
    print(" Siguiente paso: Pieza 7b (reconstruccion de la serie completa + re-QC).")
    print("=" * 70)


if __name__ == "__main__":
    main()
