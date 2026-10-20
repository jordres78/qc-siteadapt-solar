"""
qc/physical.py
--------------
Pruebas FÍSICAS del control de calidad: verifican que cada dato de
irradiancia sea físicamente posible dada la posición del sol.

Cada prueba agrega una columna de flag con 3 valores posibles:
  "OK"            -> el dato paso la prueba
  "FLAG"          -> el dato es sospechoso/imposible
  "NO_EVALUABLE"  -> la prueba no se pudo aplicar (falta el dato o la variable)

Se basa en los límites recomendados por la BSRN (físicamente posibles y
extremadamente raros). Los umbrales se definen en config.yaml (sección
qc_fisico), no en este módulo.
"""

import numpy as np
import pandas as pd

from ..config import CONFIG

QC = CONFIG["qc_fisico"]

# Etiquetas de flag (constantes para no equivocarse escribiendo)
OK = "OK"
FLAG = "FLAG"
NO_EVAL = "NO_EVALUABLE"


def _cos_sza(df):
    """Coseno del ángulo cenital solar, recortado a >= 0 (vale 0 de noche)."""
    return np.clip(np.cos(np.radians(df["zenith"])), 0, None)


# ============================================================
#  Pruebas individuales
#  Cada una recibe el DataFrame (ya con geometría solar) y el nombre
#  de la columna de irradiancia a evaluar. Devuelve una Serie de flags.
# ============================================================

def test_negativo(df, col):
    """Marca FLAG si la irradiancia es inferior al mínimo permitido (minimo_irradiancia)."""
    if col not in df.columns:
        return pd.Series(NO_EVAL, index=df.index)
    x = pd.to_numeric(df[col], errors="coerce")
    flag = np.where(x.isna(), NO_EVAL,
            np.where(x < QC["minimo_irradiancia"], FLAG, OK))
    return pd.Series(flag, index=df.index)


def test_nocturno(df, col):
    """
    Marca FLAG si hay irradiancia significativa con el sol bajo el horizonte
    (elevación <= elevacion_noche_grados e irradiancia > max_irradiancia_noche).
    Requiere la columna 'elevation' de la geometría solar.
    """
    if col not in df.columns or "elevation" not in df.columns:
        return pd.Series(NO_EVAL, index=df.index)
    x = pd.to_numeric(df[col], errors="coerce")
    es_noche = df["elevation"] <= QC["elevacion_noche_grados"]
    sospechoso = es_noche & (x > QC["max_irradiancia_noche"])
    flag = np.where(x.isna(), NO_EVAL,
            np.where(sospechoso, FLAG, OK))
    return pd.Series(flag, index=df.index)


def _limite_bsrn(df, etn_col, c1, c2, c3):
    """Límite superior BSRN: C1 * ETN * cos(SZA)^C2 + C3 (ETN: irradiancia extraterrestre normal)."""
    etn = pd.to_numeric(df[etn_col], errors="coerce")
    return c1 * etn * (_cos_sza(df) ** c2) + c3


def test_limite_fisico(df, col):
    """
    Límites físicamente posibles (PPL) de la BSRN.
    Marca FLAG si la irradiancia supera el máximo físico para esa
    posición solar o si es menor que el mínimo permitido.
    """
    if col not in df.columns or "dni_extra" not in df.columns:
        return pd.Series(NO_EVAL, index=df.index)
    x = pd.to_numeric(df[col], errors="coerce")

    if col == "GHI":
        sup = _limite_bsrn(df, "dni_extra", QC["ppl_ghi_c1"], QC["ppl_ghi_c2"], QC["ppl_ghi_c3"])
    elif col == "DHI":
        sup = _limite_bsrn(df, "dni_extra", QC["ppl_dhi_c1"], QC["ppl_dhi_c2"], QC["ppl_dhi_c3"])
    elif col == "DNI":
        sup = pd.to_numeric(df["dni_extra"], errors="coerce") * QC["ppl_dni_factor"]
    else:
        return pd.Series(NO_EVAL, index=df.index)

    fuera = (x < QC["minimo_irradiancia"]) | (x > sup)
    flag = np.where(x.isna(), NO_EVAL, np.where(fuera, FLAG, OK))
    return pd.Series(flag, index=df.index)


def test_extremely_rare(df, col):
    """
    Límites extremadamente raros (ERL) de la BSRN, más estrictos que los PPL.
    Marca FLAG los datos que, aunque físicamente posibles, son muy improbables.
    """
    if col not in df.columns or "dni_extra" not in df.columns:
        return pd.Series(NO_EVAL, index=df.index)
    x = pd.to_numeric(df[col], errors="coerce")

    if col == "GHI":
        sup = _limite_bsrn(df, "dni_extra", QC["erl_ghi_c1"], QC["erl_ghi_c2"], QC["erl_ghi_c3"])
    elif col == "DHI":
        sup = _limite_bsrn(df, "dni_extra", QC["erl_dhi_c1"], QC["erl_dhi_c2"], QC["erl_dhi_c3"])
    elif col == "DNI":
        sup = _limite_bsrn(df, "dni_extra", QC["erl_dni_c1"], QC["erl_dni_c2"], QC["erl_dni_c3"])
    else:
        return pd.Series(NO_EVAL, index=df.index)

    fuera = (x < QC["minimo_erl"]) | (x > sup)
    flag = np.where(x.isna(), NO_EVAL, np.where(fuera, FLAG, OK))
    return pd.Series(flag, index=df.index)


def test_sensor_caido(df):
    """
    Detecta sensores caídos en estaciones terrestres: TEMP=0 y RH=0 a la vez
    indica que el equipo no estaba midiendo (no es un cero físico real).
    Marca FLAG el registro; NO_EVALUABLE si la prueba está desactivada en
    config.yaml o faltan las columnas TEMP/RH.
    """
    if not QC.get("detectar_sensor_caido", False):
        return pd.Series(NO_EVAL, index=df.index)
    if "TEMP" not in df.columns or "RH" not in df.columns:
        return pd.Series(NO_EVAL, index=df.index)
    temp = pd.to_numeric(df["TEMP"], errors="coerce")
    rh = pd.to_numeric(df["RH"], errors="coerce")
    caido = (temp == 0) & (rh == 0)
    return pd.Series(np.where(caido, FLAG, OK), index=df.index)


# ============================================================
#  Orquestador: aplica todas las pruebas físicas a una fuente
# ============================================================

def aplicar_qc_fisico(df, fuente):
    """
    Aplica las pruebas físicas que correspondan según el tipo de fuente.

    fuente: un nombre que empiece con el prefijo de la fuente satelital
            (el que comprueba `es_satelital`) se trata como satelital
            -> se evalúan GHI, DNI y DHI. Se usa startswith() y no
            igualdad exacta porque puede haber varias series satelitales
            (una por caso de estudio) con etiquetas distintas que deben
            recibir las mismas pruebas.
            Cualquier otro nombre se trata como estación terrestre
            -> solo GHI + prueba de sensor caído.

    Devuelve el DataFrame con columnas de flag agregadas:
      flag_neg_GHI, flag_noche_GHI, flag_ppl_GHI, flag_erl_GHI, (idem DNI/DHI si aplica)
      flag_sensor_caido (solo terrestre)
    """
    df = df.copy()
    es_satelital = fuente.startswith("solcast")

    # Componentes a evaluar segun la fuente
    if es_satelital:
        componentes = ["GHI", "DNI", "DHI"]
    else:  # estacion terrestre
        componentes = ["GHI"]

    for col in componentes:
        df[f"flag_neg_{col}"] = test_negativo(df, col)
        df[f"flag_noche_{col}"] = test_nocturno(df, col)
        df[f"flag_ppl_{col}"] = test_limite_fisico(df, col)
        df[f"flag_erl_{col}"] = test_extremely_rare(df, col)

    if not es_satelital:
        # Todas las estaciones terrestres comparten el formato de columnas
        # normalizado, así que la prueba se aplica de la misma forma.
        df["flag_sensor_caido"] = test_sensor_caido(df)

    return df


def resumen_flags(df):
    """Cuenta los OK / FLAG / NO_EVALUABLE de cada columna de flag y el % de FLAG.
    Lanza ValueError si el DataFrame llega vacío."""
    if len(df) == 0:
        raise ValueError(
            "El DataFrame llego vacio a resumen_flags() (0 registros). "
            "Lo mas probable es que el cargador de esa fuente no haya podido "
            "leer ninguna fecha valida -- revisa el formato de la columna "
            "'date' en el CSV crudo de esa estacion."
        )
    cols_flag = [c for c in df.columns if c.startswith("flag_")]
    filas = []
    for c in cols_flag:
        conteo = df[c].value_counts()
        total = len(df)
        n_flag = int(conteo.get(FLAG, 0))
        filas.append({
            "prueba": c,
            "OK": int(conteo.get(OK, 0)),
            "FLAG": n_flag,
            "NO_EVALUABLE": int(conteo.get(NO_EVAL, 0)),
            "pct_flag": round(100 * n_flag / total, 3),
        })
    return pd.DataFrame(filas)
