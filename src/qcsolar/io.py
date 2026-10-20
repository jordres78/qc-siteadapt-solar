"""
io.py
-----
Carga y normalización de las fuentes de recurso solar.

El paquete trabaja con dos casos de estudio en un mismo sitio. Cada caso
combina una fuente terrestre con su propia serie satelital emparejada
(mismo periodo), según el mapeo `satelital_por_fuente` de config.yaml:
  - Caso 1: serie satelital (5 min, GHI/DNI/DHI y variables
    meteorológicas) + GHI terrestre estimada a partir de la potencia de
    un inversor fotovoltaico (5 min; solo columnas 'date' y 'ghi').
  - Caso 2: serie satelital del mismo píxel + estación
    meteorológica terrestre (intervalo nominal de 1 min, con huecos y
    duplicados; GHI, temperatura, humedad, viento, precipitación y
    voltaje de batería).

"Normalizar" significa dejar todas las fuentes con la misma estructura:
  - columna 'datetime' en la zona horaria local del sitio (TIMEZONE)
  - nombres de columnas estándar (GHI, DNI, DHI, TEMP, RH, ...)
  - registros ordenados en el tiempo y sin marcas de tiempo duplicadas
"""

import pandas as pd

from .config import CONFIG, TIMEZONE, resolve_path


# Mapeo de nombres originales de la fuente satelital -> nombres estándar
RENOMBRAR_SOLCAST = {
    "ghi": "GHI",
    "dni": "DNI",
    "dhi": "DHI",
    "gti": "GTI",
    "air_temp": "TEMP",
    "relative_humidity": "RH",
    "wind_speed_10m": "WS",
    "wind_direction_10m": "WD",
    "cloud_opacity": "CLOUD_OPACITY",
}

# La GHI terrestre derivada del inversor solo trae 'date' y 'ghi': al no
# haber TEMP/RH/WS, test_sensor_caido() queda NO_EVALUABLE (no marca fallo,
# simplemente no aporta esa prueba; ver qc/physical.py).
RENOMBRAR_UDENAR = {
    "ghi": "GHI",
}

# La estación meteorológica terrestre sí trae temperatura y humedad, por lo
# que test_sensor_caido() puede evaluarse sobre esta fuente.
RENOMBRAR_UDENAR_EST = {
    "irradiance": "GHI",
    "temperature": "TEMP",
    "humidity": "RH",
    "windSpeed": "WS",
    "windDirection": "WD",
    "precipitation": "PRECIP",
    "batteryVoltage": "BATTERY",
}

# Mapeo fuente terrestre -> clave de configuración de su serie satelital
# emparejada (config.yaml: satelital_por_fuente). Si una fuente no figura,
# se usa la clave de configuración por defecto.
SATELITAL_POR_FUENTE = CONFIG.get("satelital_por_fuente", {})


def cargar_solcast(clave_fuente: str | None = None) -> pd.DataFrame:
    """
    Carga y normaliza el CSV de la fuente satelital.

    `clave_fuente` es el nombre de la fuente terrestre cuya serie satelital
    emparejada se requiere; se resuelve mediante `satelital_por_fuente` en
    config.yaml. Si se omite, se usa la clave de configuración por defecto.
    """
    clave_config = SATELITAL_POR_FUENTE.get(clave_fuente, "solcast_raw") if clave_fuente else "solcast_raw"
    ruta = resolve_path(CONFIG["datos"][clave_config])
    df = pd.read_csv(ruta)

    # La columna 'period_end' ya incluye el desfase horario
    df["datetime"] = pd.to_datetime(df["period_end"], utc=True).dt.tz_convert(TIMEZONE)

    df = df.rename(columns=RENOMBRAR_SOLCAST)

    columnas = ["datetime"] + [c for c in RENOMBRAR_SOLCAST.values() if c in df.columns]
    df = df[columnas]

    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return df


def cargar_udenar() -> pd.DataFrame:
    """
    Carga y normaliza la GHI terrestre estimada a partir de la potencia del
    inversor (CSV con columnas 'date' y 'ghi', resolución de 5 min).
    """
    ruta = resolve_path(CONFIG["datos"]["udenar_raw"])
    df = pd.read_csv(ruta)

    df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(TIMEZONE, nonexistent="shift_forward", ambiguous="NaT")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(TIMEZONE)

    df = df.rename(columns=RENOMBRAR_UDENAR)
    columnas = ["datetime"] + [c for c in RENOMBRAR_UDENAR.values() if c in df.columns]
    df = df[columnas]

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return df


def cargar_udenar_estacion() -> pd.DataFrame:
    """
    Carga y normaliza el CSV de la estación meteorológica terrestre
    (columnas date, humidity, windSpeed, irradiance, temperature,
    precipitation, windDirection, batteryVoltage). El intervalo nominal es
    de 1 min, pero la serie presenta huecos y duplicados propios de una
    transmisión real, no una rejilla perfecta.
    """
    ruta = resolve_path(CONFIG["datos"]["udenar_est_raw"])
    df = pd.read_csv(ruta)

    df["datetime"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M", errors="coerce")
    df["datetime"] = df["datetime"].dt.tz_localize(TIMEZONE, nonexistent="shift_forward", ambiguous="NaT")

    df = df.rename(columns=RENOMBRAR_UDENAR_EST)
    columnas = ["datetime"] + [c for c in RENOMBRAR_UDENAR_EST.values() if c in df.columns]
    df = df[columnas]

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return df


# Cargadores de las fuentes terrestres, indexados por nombre de fuente,
# para recorrerlos en bucle desde los scripts de verificación.
CARGADORES_TERRESTRES = {
    "udenar": cargar_udenar,
    "udenar_est": cargar_udenar_estacion,
}

# Cada fuente terrestre tiene su propia serie satelital emparejada (mismo
# periodo), por lo que se procesan dos series satelitales distintas y no
# una compartida. Cada valor es una función sin argumentos, de modo que
# todos los cargadores se invocan de la misma forma: `cargador()`.
CARGADORES_SATELITALES = {
    "solcast": lambda: cargar_solcast("udenar"),             # Pareja de la GHI derivada del inversor
    "solcast_2025": lambda: cargar_solcast("udenar_est"),    # Pareja de la estación meteorológica
}

# Todas las fuentes en un solo diccionario (satelitales + terrestres), para
# aplicar el mismo procesamiento a cada una sin repetir código.
CARGADORES_TODAS = {
    **CARGADORES_SATELITALES,
    **CARGADORES_TERRESTRES,
}


def resumen_fuente(df: pd.DataFrame, nombre: str) -> dict:
    """Devuelve un resumen rápido de una fuente normalizada (registros,
    rango de fechas, intervalo mediano en minutos y variables)."""
    delta = df["datetime"].diff().dropna()
    return {
        "fuente": nombre,
        "registros": len(df),
        "fecha_inicial": df["datetime"].min(),
        "fecha_final": df["datetime"].max(),
        "intervalo_mediano_min": delta.median().total_seconds() / 60 if not delta.empty else None,
        "variables": [c for c in df.columns if c != "datetime"],
    }


# ============================================================
#  Cargadores CRUDOS (sin normalizar) - para comparaciones
#  antes/después. Devuelven los datos tal cual vienen del CSV,
#  solo con la fecha parseada para poder graficar en el tiempo.
# ============================================================

def cargar_solcast_crudo(clave_fuente: str | None = None) -> pd.DataFrame:
    """Fuente satelital tal cual viene del CSV (nombres originales, sin
    limpiar). `clave_fuente` resuelve la serie satelital emparejada con el
    mismo criterio que la carga normalizada."""
    clave_config = SATELITAL_POR_FUENTE.get(clave_fuente, "solcast_raw") if clave_fuente else "solcast_raw"
    ruta = resolve_path(CONFIG["datos"][clave_config])
    df = pd.read_csv(ruta)
    df["_fecha"] = pd.to_datetime(df["period_end"], utc=True, errors="coerce").dt.tz_convert(TIMEZONE)
    return df


def cargar_udenar_crudo() -> pd.DataFrame:
    """Fuente terrestre derivada del inversor tal cual viene del CSV (sin limpiar)."""
    ruta = resolve_path(CONFIG["datos"]["udenar_raw"])
    df = pd.read_csv(ruta)
    df["_fecha"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def cargar_udenar_estacion_crudo() -> pd.DataFrame:
    """Estación meteorológica terrestre tal cual viene del CSV (sin limpiar)."""
    ruta = resolve_path(CONFIG["datos"]["udenar_est_raw"])
    df = pd.read_csv(ruta)
    df["_fecha"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M", errors="coerce")
    return df
