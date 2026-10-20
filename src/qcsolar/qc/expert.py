"""
qc/expert.py
------------
Pruebas EXPERTAS y de serie temporal del control de calidad. Van más allá
de los límites físicos: usan la relación entre componentes y el
comportamiento temporal de la serie.

Familias de pruebas:
  - Cierre radiativo  : GHI ~ DNI*cos(SZA) + DHI   (solo satelital, 3 componentes)
  - K-tests           : índices de claridad Kt, Kn, K (solo satelital)
  - CSI               : índice de cielo claro (ambas fuentes)
  - Rate of change    : saltos imposibles entre muestras (ambas)
  - Persistencia      : valores congelados, 15 valores idénticos consecutivos (ambas)
  - Simetría AM/PM    : asimetría mañana/tarde por día (ambas; solo diagnóstico)

Cada prueba agrega una columna de flag con: OK / FLAG / NO_EVALUABLE.
Los umbrales se definen en config.yaml (sección qc_experto), siguiendo el
esquema de QC experto para datos terrestres de radiación solar.
"""

import numpy as np
import pandas as pd

from ..config import CONFIG

QE = CONFIG["qc_experto"]
ALT = CONFIG["sitio"]["altitud_m"]

OK = "OK"
FLAG = "FLAG"
NO_EVAL = "NO_EVALUABLE"


def _num(df, col):
    """Convierte la columna `col` a numérica (valores no convertibles -> NaN)."""
    return pd.to_numeric(df[col], errors="coerce")


def _cos_sza(df):
    """Coseno del ángulo cenital solar, recortado a >= 0 (vale 0 de noche)."""
    return np.clip(np.cos(np.radians(df["zenith"])), 0, None)


# ============================================================
#  1. Cierre radiativo (solo fuente satelital)
# ============================================================

def test_cierre(df):
    """
    GHI debe coincidir con DNI*cos(SZA) + DHI dentro de una tolerancia
    que depende del ángulo cenital. Detecta inconsistencias entre las
    tres componentes; solo se evalúa con GHI > cierre_ghi_min y SZA < 90.
    """
    req = {"GHI", "DNI", "DHI", "zenith"}
    if not req.issubset(df.columns):
        return pd.Series(NO_EVAL, index=df.index)

    ghi = _num(df, "GHI")
    dni = _num(df, "DNI")
    dhi = _num(df, "DHI")
    sza = df["zenith"]
    ghi_calc = dni * _cos_sza(df) + dhi

    # Tolerancia según SZA (más estricta con el sol alto)
    tol = np.where(sza <= QE["cierre_sza_corte"],
                   QE["cierre_tol_sza_baja"], QE["cierre_tol_sza_alta"])

    with np.errstate(divide="ignore", invalid="ignore"):
        desvio = np.abs(ghi / ghi_calc - 1)

    evaluable = (ghi > QE["cierre_ghi_min"]) & (ghi_calc > 0) & (sza < 90)
    flag = np.where(~evaluable | ghi.isna(), NO_EVAL,
            np.where(desvio > tol, FLAG, OK))
    return pd.Series(flag, index=df.index)


# ============================================================
#  2. K-tests (solo fuente satelital)
# ============================================================

def test_ktests(df):
    """
    Índices de claridad y sus límites físicos:
      Kt = GHI/(ETN*cos SZA) < kt_max
      Kn = DNI/ETN < (kn_factor + kn_alt_coef*ALT)/ETN
      K  = DHI/GHI  con límites según SZA (y uno adicional para Kt alto)
    Devuelve FLAG si ALGUNO de los índices viola su límite.
    """
    req = {"GHI", "DNI", "DHI", "zenith", "dni_extra"}
    if not req.issubset(df.columns):
        return pd.Series(NO_EVAL, index=df.index)

    ghi = _num(df, "GHI")
    dni = _num(df, "DNI")
    dhi = _num(df, "DHI")
    etn = _num(df, "dni_extra")
    cos = _cos_sza(df)
    sza = df["zenith"]

    with np.errstate(divide="ignore", invalid="ignore"):
        kt = ghi / (etn * cos)
        kn = dni / etn
        k = dhi / ghi

    limite_kn = (QE["kn_factor"] + QE["kn_alt_coef"] * ALT) / etn
    k_max = np.where(sza < QE["cierre_sza_corte"], QE["k_max_sza_baja"], QE["k_max_sza_alta"])

    viola = (
        (kt > QE["kt_max"]) |
        (kn > limite_kn) |
        (k > k_max) |
        ((kt > QE["k_kt_corte"]) & (ghi > QE["kkt_ghi_min"]) &
         (sza < QE["kkt_sza_max"]) & (k > QE["k_max_kkt"]))
    )

    evaluable = (ghi > QE["ktests_ghi_min"]) & (etn * cos > 0)
    flag = np.where(~evaluable | ghi.isna(), NO_EVAL,
            np.where(viola, FLAG, OK))
    return pd.Series(flag, index=df.index)


# ============================================================
#  3. CSI - índice de cielo claro (ambas fuentes)
# ============================================================

def test_csi(df):
    """
    CSI = GHI_medido / GHI_clearsky. Bajo cielo claro debería ser ~1.
    Marca FLAG si el CSI supera csi_max (dato físicamente dudoso).
    Los valores bajos sostenidos (p. ej. sensor sucio) no se marcan aquí;
    se analizan aparte de forma gráfica.
    """
    if "GHI" not in df.columns or "ghi_clear" not in df.columns:
        return pd.Series(NO_EVAL, index=df.index)
    ghi = _num(df, "GHI")
    ghi_clear = _num(df, "ghi_clear")
    with np.errstate(divide="ignore", invalid="ignore"):
        csi = ghi / ghi_clear
    evaluable = (ghi > QE["csi_ghi_min"]) & (ghi_clear > QE["csi_ghi_min"])
    flag = np.where(~evaluable | ghi.isna(), NO_EVAL,
            np.where(csi > QE["csi_max"], FLAG, OK))
    return pd.Series(flag, index=df.index)


# ============================================================
#  4. Rate of change (ambas fuentes)
# ============================================================

def test_rate_of_change(df, col="GHI"):
    """
    Marca FLAG si el salto absoluto de la variable entre dos muestras
    consecutivas supera el máximo plausible (roc_max_wm2). Detecta picos y
    errores súbitos.
    """
    if col not in df.columns:
        return pd.Series(NO_EVAL, index=df.index)
    x = _num(df, col)
    salto = x.diff().abs()
    flag = np.where(salto.isna(), NO_EVAL,
            np.where(salto > QE["roc_max_wm2"], FLAG, OK))
    return pd.Series(flag, index=df.index)


# ============================================================
#  5. Persistencia (ambas fuentes)
# ============================================================

def test_persistencia(df, col="GHI"):
    """
    Marca FLAG si un mismo valor se repite N o más veces consecutivas
    (N = persistencia_n, 15 por defecto) MIENTRAS EL SOL ESTÁ SOBRE EL
    HORIZONTE: señal de sensor o datalogger congelado ("log-down"), sin
    importar si el valor repetido es alto o bajo. Un corte de energía o una
    falla de sensor puede congelar la lectura en CUALQUIER valor, incluido
    cero: una irradiancia nula durante horas de pleno día (sol confirmado
    por la geometría solar) es tan sospechosa como un valor alto congelado,
    por eso no se exige un piso de magnitud al valor repetido.

    De noche la prueba NO se evalúa (NO_EVALUABLE): allí un GHI=0
    sostenido es físicamente normal. El corte día/noche usa el mismo
    umbral qc_fisico.elevacion_noche_grados que test_nocturno() en
    physical.py, por consistencia.

    N=15 sin piso de magnitud sigue la definición estándar de la prueba de
    persistencia del QC experto; no es un umbral calibrado a un sitio.
    """
    if col not in df.columns or "elevation" not in df.columns:
        return pd.Series(NO_EVAL, index=df.index)
    x = _num(df, col)
    es_de_dia = df["elevation"] > CONFIG["qc_fisico"]["elevacion_noche_grados"]
    # Identificar rachas de valores idénticos consecutivos
    cambio = (x != x.shift()).cumsum()
    largo_racha = x.groupby(cambio).transform("size")
    congelado = (largo_racha >= QE["persistencia_n"]) & es_de_dia
    flag = np.where(x.isna() | ~es_de_dia, NO_EVAL,
            np.where(congelado, FLAG, OK))
    return pd.Series(flag, index=df.index)


# ============================================================
#  6. Simetría AM/PM (ambas fuentes) - diagnóstico por día
# ============================================================

def test_simetria_ampm(df, col="GHI"):
    """
    Por cada día compara la energía de la mañana (antes de las 12 h locales)
    con la de la tarde. Una asimetría grande y sostenida puede indicar mala
    nivelación del sensor o error de hora. El flag se asigna a TODO el día.
    Es una prueba solo diagnóstica: no interviene en la regla de usabilidad
    (ver PRUEBAS_SOLO_DIAGNOSTICO en qc/usability.py).
    """
    if col not in df.columns or "elevation" not in df.columns:
        return pd.Series(NO_EVAL, index=df.index)

    d = df[["datetime", col, "elevation"]].copy()
    d[col] = _num(d, col)
    d["fecha"] = d["datetime"].dt.date
    d["am"] = d["datetime"].dt.hour < 12  # mañana vs tarde (hora local, aproximación del mediodía solar)
    d["dia"] = d["elevation"] > 0

    resultado = pd.Series(OK, index=df.index)
    for fecha, grupo in d[d["dia"]].groupby("fecha"):
        am = grupo[grupo["am"]][col].sum()
        pm = grupo[~grupo["am"]][col].sum()
        if (am + pm) <= 0:
            continue
        asimetria = abs(am - pm) / (am + pm)
        if asimetria > QE["simetria_tol"]:
            resultado.loc[grupo.index] = FLAG
    # Las filas nocturnas quedan NO_EVALUABLE
    resultado.loc[~d["dia"]] = NO_EVAL
    return resultado


# ============================================================
#  Orquestador
# ============================================================

def aplicar_qc_experto(df, fuente):
    """
    Aplica las pruebas expertas según el tipo de fuente.
      satelital -> cierre, K-tests, CSI, ROC, persistencia, simetría
      terrestre -> CSI, ROC, persistencia, simetría (no tiene DNI/DHI)

    `fuente` se clasifica por prefijo (startswith) y no por igualdad
    exacta, igual que en qc/physical.py, porque puede haber varias series
    satelitales (una por caso de estudio) con etiquetas distintas.
    """
    df = df.copy()

    if fuente.startswith("solcast"):
        df["flag_cierre"] = test_cierre(df)
        df["flag_ktests"] = test_ktests(df)

    df["flag_csi"] = test_csi(df)
    df["flag_roc_GHI"] = test_rate_of_change(df, "GHI")
    df["flag_persistencia_GHI"] = test_persistencia(df, "GHI")
    df["flag_simetria_GHI"] = test_simetria_ampm(df, "GHI")

    return df


def resumen_flags_experto(df):
    """Cuenta OK/FLAG/NO_EVALUABLE y el % de FLAG de las columnas de flag experto."""
    cols = [c for c in df.columns if c.startswith("flag_") and
            any(k in c for k in ["cierre", "ktests", "csi", "roc", "persistencia", "simetria"])]
    filas = []
    for c in cols:
        vc = df[c].value_counts()
        n_flag = int(vc.get(FLAG, 0))
        filas.append({
            "prueba": c,
            "OK": int(vc.get(OK, 0)),
            "FLAG": n_flag,
            "NO_EVALUABLE": int(vc.get(NO_EVAL, 0)),
            "pct_flag": round(100 * n_flag / len(df), 3),
        })
    return pd.DataFrame(filas)
