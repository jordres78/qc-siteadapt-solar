"""
siteadapt/prepare.py
---------------------
Preparación de datos para el site-adaptation: emparejamiento de la serie
satelital (predictoras) con la estación terrestre (objetivo) y partición
temporal en entrenamiento/prueba.

Responsabilidades, separadas para poder probarlas una a la vez:

1. `emparejar_horario_generico()`: empareja la fuente satelital con la
   terrestre a la resolución indicada (5 min en el flujo principal),
   usando solo registros que pasaron la regla de usabilidad, y devuelve
   el conjunto supervisado (entrenar/evaluar) y el conjunto de aplicación
   (reconstruir la serie completa). `emparejar_horario()` ofrece el
   emparejamiento horario básico, sin conjunto de aplicación.

2. `split_temporal()`: divide el conjunto emparejado en train/test de
   forma TEMPORAL (no aleatoria): el primer 70 % de las fechas para
   entrenar y el último 30 % para evaluar. Una partición aleatoria (o un
   k-fold aleatorio) mezclaría muestras contiguas, fuertemente
   autocorrelacionadas, entre train y test y produciría fuga de datos.

El periodo de análisis (periodo_inicio / periodo_fin) se recibe como
parámetro explícito o se toma de config.yaml (sección site_adaptation).
"""

import numpy as np
import pandas as pd

from ..config import CONFIG


# Variables de la fuente satelital candidatas a predictoras de los modelos
VARIABLES_SOLCAST_PREDICTORAS = [
    "GHI", "DNI", "DHI", "GTI", "TEMP", "RH", "WS", "CLOUD_OPACITY",
    "zenith", "elevation",
]


def emparejar_horario(df_solcast: pd.DataFrame, df_alvernia: pd.DataFrame,
                       periodo_inicio: str | None = None,
                       periodo_fin: str | None = None,
                       columna_usabilidad: str = "usabilidad",
                       cobertura_minima: float | None = None,
                       variables_solcast: list[str] | None = None) -> pd.DataFrame:
    """
    Empareja la fuente satelital (primer argumento) y la estación terrestre
    (segundo argumento) a resolución horaria.

    Requisitos de ambos DataFrames:
      - Tener la columna 'datetime' (con zona horaria).
      - Tener la columna `columna_usabilidad` con valores "OK"/"NO_USABLE"
        (geometría solar + QC físico + QC experto + regla de usabilidad).

    Una hora solo entra al resultado si, en AMBAS fuentes, la fracción de
    registros OK sobre lo esperado en esa hora es >= `cobertura_minima`
    (0.8 por defecto: mínimo 80 % de cobertura). Así una hora con un solo
    dato OK de los 12 esperados no cuenta igual que una hora con cobertura
    casi completa.

    De la fuente satelital se promedian todas las variables indicadas en
    el último parámetro (por defecto, la lista de predictoras candidatas). De la
    estación terrestre solo se promedia GHI: es la variable objetivo, y sus
    demás variables no se usan como predictoras porque en operación solo
    se dispone de la fuente satelital.

    También agrega columnas temporales derivadas (hour, dayofyear, month),
    útiles como predictoras para los modelos basados en árboles.

    Devuelve un DataFrame con: datetime, las variables satelitales
    promediadas, la GHI terrestre, los conteos n_* y las coberturas
    cobertura_* de cada fuente, y hour, dayofyear, month.
    """
    if periodo_inicio is None:
        periodo_inicio = CONFIG["site_adaptation"]["periodo_inicio"]
    if periodo_fin is None:
        periodo_fin = CONFIG["site_adaptation"]["periodo_fin"]
    if cobertura_minima is None:
        cobertura_minima = CONFIG["site_adaptation"].get("cobertura_minima_horaria", 0.8)
    if variables_solcast is None:
        variables_solcast = VARIABLES_SOLCAST_PREDICTORAS

    tz = df_solcast["datetime"].dt.tz
    inicio = pd.Timestamp(periodo_inicio, tz=tz)
    fin = pd.Timestamp(periodo_fin, tz=tz) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    def _promedio_horario_ok(df, nombre_col, variables, muestras_esperadas_por_hora=None):
        """Promedia por hora las `variables` OK de `df` dentro del periodo y
        calcula conteo y cobertura; devuelve (agregado, muestras esperadas)."""
        d = df[(df["datetime"] >= inicio) & (df["datetime"] <= fin)].copy()
        d = d[d[columna_usabilidad] == "OK"]
        variables_presentes = [v for v in variables if v in d.columns]
        d = d.set_index("datetime")[variables_presentes]

        agregado = d.resample("1h").mean()
        agregado.columns = [f"{c}_{nombre_col}" for c in agregado.columns]
        agregado[f"n_{nombre_col}"] = d.iloc[:, 0].resample("1h").size() if len(variables_presentes) else 0

        if muestras_esperadas_por_hora is None:
            conteos_no_cero = agregado[f"n_{nombre_col}"][agregado[f"n_{nombre_col}"] > 0]
            muestras_esperadas_por_hora = conteos_no_cero.median() if len(conteos_no_cero) else np.nan

        agregado[f"cobertura_{nombre_col}"] = agregado[f"n_{nombre_col}"] / muestras_esperadas_por_hora
        return agregado, muestras_esperadas_por_hora

    # La fuente satelital tiene paso fijo de 5 min -> 12 muestras esperadas por hora.
    sol_h, _ = _promedio_horario_ok(df_solcast, "solcast", variables_solcast, muestras_esperadas_por_hora=12)
    alv_h, alvernia_esperadas = _promedio_horario_ok(df_alvernia, "alvernia", ["GHI"])

    pareado = sol_h.join(alv_h, how="inner")
    pareado = pareado[
        (pareado["cobertura_solcast"] >= cobertura_minima) &
        (pareado["cobertura_alvernia"] >= cobertura_minima)
    ]
    pareado = pareado.dropna(subset=["GHI_solcast", "GHI_alvernia"])
    pareado = pareado.reset_index().rename(columns={"index": "datetime"})

    pareado["hour"] = pareado["datetime"].dt.hour
    pareado["dayofyear"] = pareado["datetime"].dt.dayofyear
    pareado["month"] = pareado["datetime"].dt.month

    return pareado


def split_temporal(df_pareado: pd.DataFrame, fraccion_train: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el conjunto emparejado en train/test de forma TEMPORAL: la
    primera fracción `fraccion_train` de las muestras (ordenadas por fecha,
    no al azar) va a train y el resto a test.

    Devuelve (train, test). Las métricas de comparación entre modelos se
    calculan solo sobre `test`.
    """
    if fraccion_train is None:
        fraccion_train = CONFIG["site_adaptation"]["fraccion_train"]

    df_pareado = df_pareado.sort_values("datetime").reset_index(drop=True)
    corte = int(len(df_pareado) * fraccion_train)
    train = df_pareado.iloc[:corte].reset_index(drop=True)
    test = df_pareado.iloc[corte:].reset_index(drop=True)
    return train, test


def resumen_emparejamiento(df_pareado: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Resumen rápido (tamaños, rangos de fechas, sesgo y correlación antes de
    corregir) para verificar el emparejamiento y la partición."""
    return {
        "horas_emparejadas": len(df_pareado),
        "rango_fechas": (str(df_pareado["datetime"].min()), str(df_pareado["datetime"].max())),
        "n_train": len(train),
        "rango_train": (str(train["datetime"].min()), str(train["datetime"].max())) if len(train) else None,
        "n_test": len(test),
        "rango_test": (str(test["datetime"].min()), str(test["datetime"].max())) if len(test) else None,
        "sesgo_medio_antes (Solcast-Alvernia, W/m2)": round(
            (df_pareado["GHI_solcast"] - df_pareado["GHI_alvernia"]).mean(), 2),
        "correlacion_antes": round(df_pareado["GHI_solcast"].corr(df_pareado["GHI_alvernia"]), 3),
    }


# ================================================================
#  RECONSTRUCCIÓN DE LA SERIE TERRESTRE: la fuente satelital actúa
#  como referencia (predictoras) para corregir y rellenar huecos en la
#  fuente terrestre (objetivo). Además del conjunto supervisado, se
#  construye el conjunto de aplicación sobre todo el periodo disponible.
# ================================================================

def _promedio_horario(df: pd.DataFrame, sufijo: str, variables: list[str],
                       filtrar_usabilidad: bool,
                       columna_usabilidad: str = "usabilidad",
                       muestras_esperadas: float | None = None,
                       freq: str = "1h"):
    """
    Promedia `variables` de `df` a resolución `freq` (1h por defecto;
    también acepta p. ej. "5min" para trabajar a resolución nativa cuando
    la fuente viene muestreada cada 5 min), indexado por datetime.
    Si `filtrar_usabilidad=True`, solo usa filas con columna_usabilidad=="OK".
    Si es False, promedia todo lo que haya en cada bin sin importar la
    bandera, para distinguir un dato existente pero marcado de un hueco
    real.

    A resolución nativa (freq == intervalo real de muestreo) cada bin
    contiene como máximo 1 registro, así que la cobertura se vuelve binaria
    (1.0 si hay dato OK, 0.0 si no); el mecanismo es el mismo, solo cambia
    la granularidad del bin.

    Devuelve (df_agregado, muestras_esperadas_por_bin). Columnas del
    resultado: "{var}_{sufijo}" por cada variable, "n_{sufijo}" (conteo
    de registros promediados en ese bin) y "cobertura_{sufijo}" (n / muestras
    esperadas).
    """
    d = df.copy()
    if filtrar_usabilidad:
        d = d[d[columna_usabilidad] == "OK"]
    variables_presentes = [v for v in variables if v in d.columns]
    d = d.set_index("datetime")[variables_presentes]
    if filtrar_usabilidad and len(variables_presentes):
        # Protección: un dato crudo faltante (NaN) hace que TODAS las pruebas
        # de qc.physical/qc.expert devuelvan NO_EVALUABLE (ninguna FLAG), y
        # calcular_usabilidad() lo clasifica entonces como "OK" por ausencia
        # de FLAG aunque el dato no exista. A resolución horaria el promedio
        # ignora ese NaN si hay otras muestras válidas; a resolución nativa
        # (1 muestra por bin) ese NaN sería el valor "medido_ok" completo
        # del bin e impediría el entrenamiento (sklearn no acepta NaN en y).
        # Por eso se descarta aquí cualquier fila "OK" sin dato.
        d = d.dropna(how="any")

    agregado = d.resample(freq).mean()
    agregado.columns = [f"{c}_{sufijo}" for c in agregado.columns]
    agregado[f"n_{sufijo}"] = d.iloc[:, 0].resample(freq).size() if len(variables_presentes) else 0

    if muestras_esperadas is None:
        conteos_no_cero = agregado[f"n_{sufijo}"][agregado[f"n_{sufijo}"] > 0]
        muestras_esperadas = conteos_no_cero.median() if len(conteos_no_cero) else float("nan")

    agregado[f"cobertura_{sufijo}"] = agregado[f"n_{sufijo}"] / muestras_esperadas
    return agregado, muestras_esperadas


def emparejar_horario_generico(
    df_referencia: pd.DataFrame,
    df_objetivo: pd.DataFrame,
    nombre_objetivo: str,
    col_objetivo: str = "GHI",
    periodo_inicio: str | None = None,
    periodo_fin: str | None = None,
    columna_usabilidad: str = "usabilidad",
    cobertura_minima: float | None = None,
    variables_referencia: list[str] | None = None,
    excluir_desde: str | None = None,
    excluir_hasta: str | None = None,
    muestras_esperadas_referencia: float | None = None,
    freq: str = "1h",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Empareja la REFERENCIA (fuente satelital, completa y consistente, que
    aporta las predictoras) con el OBJETIVO (estación terrestre, cuya GHI
    se quiere corregir y reconstruir). Devuelve DOS conjuntos:

    1. `supervisado` -- bins donde referencia Y objetivo tienen cobertura
       >= cobertura_minima de datos usabilidad=="OK", dentro de
       [periodo_inicio, periodo_fin] si se dan, y FUERA de la ventana
       [excluir_desde, excluir_hasta] si se da (p. ej. un periodo con una
       falla conocida del sensor terrestre). Es el único conjunto que se
       usa para ENTRENAR y EVALUAR modelos: la verdad de terreno confiable.

    2. `aplicacion` -- TODOS los bins donde la referencia tiene cobertura
       suficiente, en todo el rango disponible (sin recortar por periodo ni
       exclusiones). Incluye una columna `origen_dato` por bin:
         "medido_ok"   -> el objetivo tiene dato usable en ese bin (está en
                          `supervisado` o lo estaría de no caer en la
                          ventana excluida)
         "medido_flag" -> el objetivo tiene algún registro en ese bin, pero
                          no alcanza cobertura/usabilidad OK
         "sin_dato"    -> el objetivo no tiene NINGÚN registro en ese bin
       y `GHI_{nombre_objetivo}_crudo` con el promedio crudo (sin filtrar
       por usabilidad) de col_objetivo, NaN si "sin_dato". Sobre este
       conjunto se aplica el modelo ganador para reconstruir la serie
       completa de la estación (variantes total y selectiva).

    `freq` fija la resolución de emparejamiento: "1h" por defecto o una
    resolución nativa como "5min" (la usada en el flujo principal), que
    da más puntos de entrenamiento y aplicación a costa de un objetivo más
    ruidoso (variabilidad instantánea de nubes en vez de promedio
    horario). Si no se pasa `muestras_esperadas_referencia`, se calcula
    como el número de muestras de 5 min que caben en `freq` (12 para
    "1h", 1 para "5min").

    Ambos DataFrames comparten las mismas columnas predictoras (variables
    satelitales con su sufijo y temporales hour, dayofyear, month), listas
    para pasarlas a `evaluate.entrenar_y_evaluar()`.
    """
    if cobertura_minima is None:
        cobertura_minima = CONFIG["site_adaptation"].get("cobertura_minima_horaria", 0.8)
    if variables_referencia is None:
        variables_referencia = VARIABLES_SOLCAST_PREDICTORAS
    if muestras_esperadas_referencia is None:
        # Cuántas muestras nativas de 5 min caben en un bin de `freq`
        # (12 para "1h", 1 para "5min", etc.); asume que la fuente de
        # referencia viene muestreada cada 5 min.
        muestras_esperadas_referencia = max(1, round(pd.Timedelta(freq) / pd.Timedelta("5min")))

    # --- Referencia (satelital): promedio a resolución `freq`, solo datos OK ---
    ref_h, _ = _promedio_horario(
        df_referencia, "solcast", variables_referencia,
        filtrar_usabilidad=True, columna_usabilidad=columna_usabilidad,
        muestras_esperadas=muestras_esperadas_referencia, freq=freq,
    )
    ref_valida = ref_h[ref_h["cobertura_solcast"] >= cobertura_minima].copy()

    # --- Objetivo (estación terrestre): DOS agregaciones ---
    # (a) solo datos OK: para saber si el bin califica como "medido_ok"
    obj_ok_h, obj_esperadas = _promedio_horario(
        df_objetivo, "obj_ok", [col_objetivo],
        filtrar_usabilidad=True, columna_usabilidad=columna_usabilidad, freq=freq,
    )
    # (b) todo lo que haya, sin filtrar por usabilidad: para distinguir
    #     "había dato pero marcado" de "hueco real sin ningún registro"
    obj_bruto_h, _ = _promedio_horario(
        df_objetivo, "obj_bruto", [col_objetivo],
        filtrar_usabilidad=False, muestras_esperadas=obj_esperadas, freq=freq,
    )

    aplicacion = ref_valida.join(obj_bruto_h, how="left").join(obj_ok_h, how="left")

    def _clasificar(fila):
        """Asigna el origen del dato del bin: medido_ok, medido_flag o sin_dato."""
        cob_ok = fila.get("cobertura_obj_ok")
        if pd.notna(cob_ok) and cob_ok >= cobertura_minima:
            return "medido_ok"
        n_bruto = fila.get("n_obj_bruto")
        if pd.notna(n_bruto) and n_bruto > 0:
            return "medido_flag"
        return "sin_dato"

    aplicacion["origen_dato"] = aplicacion.apply(_clasificar, axis=1)
    aplicacion = aplicacion.rename(columns={
        f"{col_objetivo}_obj_bruto": f"GHI_{nombre_objetivo}_crudo",
        f"{col_objetivo}_obj_ok": f"GHI_{nombre_objetivo}",
    })
    aplicacion = aplicacion.reset_index().rename(columns={"index": "datetime"})
    aplicacion["hour"] = aplicacion["datetime"].dt.hour
    aplicacion["dayofyear"] = aplicacion["datetime"].dt.dayofyear
    aplicacion["month"] = aplicacion["datetime"].dt.month

    # --- Supervisado: subconjunto "medido_ok" de aplicación, recortado
    #     por periodo y exclusiones (verdad de terreno confiable) ---
    supervisado = aplicacion[aplicacion["origen_dato"] == "medido_ok"].copy()

    tz = supervisado["datetime"].dt.tz
    if periodo_inicio is not None:
        inicio = pd.Timestamp(periodo_inicio, tz=tz)
        supervisado = supervisado[supervisado["datetime"] >= inicio]
    if periodo_fin is not None:
        fin = pd.Timestamp(periodo_fin, tz=tz) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        supervisado = supervisado[supervisado["datetime"] <= fin]
    if excluir_desde is not None:
        desde = pd.Timestamp(excluir_desde, tz=tz)
        # "2100-01-01" en vez de pd.Timestamp.max: el máximo de pandas
        # (nanosegundos) se desborda al localizarlo en zonas horarias con
        # desfase negativo (p. ej. UTC-05:00); 2100 basta para cualquier
        # serie de medición real.
        hasta = pd.Timestamp(excluir_hasta, tz=tz) if excluir_hasta is not None else pd.Timestamp("2100-01-01", tz=tz)
        supervisado = supervisado[~((supervisado["datetime"] >= desde) & (supervisado["datetime"] <= hasta))]

    supervisado = supervisado.reset_index(drop=True)
    aplicacion = aplicacion.sort_values("datetime").reset_index(drop=True)

    return supervisado, aplicacion


def resumen_emparejamiento_generico(supervisado: pd.DataFrame, aplicacion: pd.DataFrame,
                                     nombre_objetivo: str, train: pd.DataFrame,
                                     test: pd.DataFrame) -> dict:
    """Resumen del emparejamiento genérico para una estación objetivo: tamaños
    y rangos de cada conjunto, conteo por origen_dato, sesgo y correlación
    antes de corregir."""
    col_obj = f"GHI_{nombre_objetivo}"
    conteo_origen = aplicacion["origen_dato"].value_counts().to_dict()
    return {
        "estacion": nombre_objetivo,
        # Las claves se llaman "muestras_..." (no "horas_...") porque el
        # emparejamiento se hace a resolución nativa de 5 min: estos conteos
        # son ~12 veces mayores que un conteo horario equivalente.
        "muestras_emparejadas_supervisado": len(supervisado),
        "rango_fechas_supervisado": (
            (str(supervisado["datetime"].min()), str(supervisado["datetime"].max()))
            if len(supervisado) else None
        ),
        "n_train": len(train),
        "rango_train": (str(train["datetime"].min()), str(train["datetime"].max())) if len(train) else None,
        "n_test": len(test),
        "rango_test": (str(test["datetime"].min()), str(test["datetime"].max())) if len(test) else None,
        "muestras_aplicacion_total": len(aplicacion),
        "muestras_medido_ok": int(conteo_origen.get("medido_ok", 0)),
        "muestras_medido_flag": int(conteo_origen.get("medido_flag", 0)),
        "muestras_sin_dato": int(conteo_origen.get("sin_dato", 0)),
        "sesgo_medio_antes (Solcast-estacion, W/m2)": round(
            (supervisado["GHI_solcast"] - supervisado[col_obj]).mean(), 2) if len(supervisado) else None,
        "correlacion_antes": round(
            supervisado["GHI_solcast"].corr(supervisado[col_obj]), 3) if len(supervisado) else None,
    }
