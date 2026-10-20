"""
solar.py
--------
Geometria solar: a partir de un datetime y la ubicacion del sitio
(definida en config.yaml), calcula la posicion del sol en cada instante.
Es la base de casi todas las pruebas de control de calidad, porque
permite comparar el dato medido contra lo fisicamente posible en ese
momento.

Se apoya en la libreria pvlib (estandar en la industria solar).

Conceptos que calcula:
  - zenith    : angulo entre el sol y la vertical. 90 grados = sol en
                el horizonte; >90 = sol bajo el horizonte (de noche).
  - elevation : que tan alto esta el sol sobre el horizonte (90 - zenith).
  - azimuth   : hacia que punto cardinal esta el sol (0=Norte, 180=Sur).
  - es_de_dia : True si el sol esta sobre el horizonte.
  - dni_extra : irradiancia que llega ARRIBA de la atmosfera (limite teorico).
  - ghi_clear : GHI que habria con cielo COMPLETAMENTE despejado (modelo).
  - dni_clear, dhi_clear : idem para las otras componentes.
"""

import numpy as np
import pandas as pd
import pvlib

from .config import LATITUD, LONGITUD, ALTITUD_M, TIMEZONE


def agregar_geometria_solar(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """
    Agrega columnas de geometria solar y cielo claro a un DataFrame.

    La columna `datetime_col` debe idealmente tener zona horaria (asi la
    dejan los cargadores de io.py); si no la tiene, se localiza en la zona
    horaria del sitio.

    Devuelve una copia del DataFrame con columnas nuevas:
      zenith, elevation, azimuth, es_de_dia, es_de_noche,
      dni_extra, ghi_clear, dni_clear, dhi_clear
    """
    df = df.copy()

    # pvlib requiere un indice temporal con zona horaria.
    tiempos = pd.DatetimeIndex(df[datetime_col])
    if tiempos.tz is None:
        tiempos = tiempos.tz_localize(TIMEZONE)
    else:
        tiempos = tiempos.tz_convert(TIMEZONE)

    # --- Posicion del sol ---
    posicion = pvlib.solarposition.get_solarposition(
        tiempos, LATITUD, LONGITUD, altitude=ALTITUD_M
    )
    df["zenith"] = posicion["zenith"].to_numpy()
    df["elevation"] = posicion["elevation"].to_numpy()
    df["azimuth"] = posicion["azimuth"].to_numpy()

    # --- Dia / noche ---
    # Se usa elevacion > 0: el sol esta sobre el horizonte.
    df["es_de_dia"] = df["elevation"] > 0
    df["es_de_noche"] = df["zenith"] > 90  # equivalente, perspectiva opuesta

    # --- Irradiancia extraterrestre (limite teorico superior) ---
    # Es la irradiancia que llega arriba de la atmosfera.
    df["dni_extra"] = pvlib.irradiance.get_extra_radiation(tiempos).to_numpy()

    # --- Modelo de cielo despejado (clear-sky, Ineichen) ---
    # Irradiancia esperada con cielo TOTALMENTE limpio. Sirve de
    # referencia: el dato real no deberia superarlo de forma sostenida.
    sitio = pvlib.location.Location(
        LATITUD, LONGITUD, tz=TIMEZONE, altitude=ALTITUD_M, name="Pasto"
    )
    cielo_claro = sitio.get_clearsky(tiempos, model="ineichen")
    df["ghi_clear"] = cielo_claro["ghi"].to_numpy()
    df["dni_clear"] = cielo_claro["dni"].to_numpy()
    df["dhi_clear"] = cielo_claro["dhi"].to_numpy()

    return df


def resumen_geometria(df: pd.DataFrame) -> dict:
    """Resumen rapido (registros, conteos dia/noche, elevacion maxima,
    GHI clear-sky maximo y ETN promedio) para verificar la geometria."""
    return {
        "registros": len(df),
        "horas_de_dia": int(df["es_de_dia"].sum()),
        "horas_de_noche": int(df["es_de_noche"].sum()),
        "elevacion_max_grados": round(float(df["elevation"].max()), 2),
        "ghi_clear_max": round(float(df["ghi_clear"].max()), 1),
        "dni_extra_promedio": round(float(df["dni_extra"].mean()), 1),
    }
