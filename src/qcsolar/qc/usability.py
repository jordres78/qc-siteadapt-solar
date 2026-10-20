"""
qc/usability.py
----------------
Regla de usabilidad: combina las columnas flag_* (físicas + expertas) en
un solo veredicto por marca de tiempo.

Regla base: si al menos una prueba EVALUABLE (y no solo diagnóstica) marcó
FLAG, el dato se considera NO_USABLE. Las pruebas NO_EVALUABLE no cuentan
ni a favor ni en contra: simplemente no opinan sobre ese dato.

Regla de islas ("data islands"): los segmentos usables aislados entre
datos NO_USABLE, de menos de 60 min, también se descartan, para no dejar
puntos buenos sueltos en medio de un tramo malo. Se implementa en
`aplicar_regla_islas()`, separada de la regla base, para poder cuantificar
el efecto de cada una por separado.
"""

import numpy as np
import pandas as pd

OK = "OK"
NO_USABLE = "NO_USABLE"

# Pruebas de "inspección visual": sirven como apoyo diagnóstico para que un
# experto decida caso por caso (p. ej. asimetría mañana/tarde por nubosidad
# real vs. error de sensor), NO como regla automática de exclusión. Se
# calculan y reportan (ver test_simetria_ampm en qc/expert.py), pero no
# descartan por sí solas una marca de tiempo o un día completo dentro de
# calcular_usabilidad().
PRUEBAS_SOLO_DIAGNOSTICO = ["simetria"]


def calcular_usabilidad(df: pd.DataFrame, columnas_flag: list[str] | None = None) -> pd.Series:
    """
    Combina las columnas de flag_* en un veredicto por fila:
      "OK"        -> ninguna prueba evaluable (y no diagnóstica) marcó FLAG
      "NO_USABLE" -> al menos una prueba marcó FLAG

    Si no se pasa columnas_flag, usa TODAS las columnas que empiecen con
    "flag_" presentes en el DataFrame (resultado de aplicar_qc_fisico() y
    aplicar_qc_experto()), EXCEPTO las de solo diagnóstico (ver
    PRUEBAS_SOLO_DIAGNOSTICO). El filtro se aplica también cuando se pasa
    columnas_flag explícitamente, para que una prueba diagnóstica nunca
    actúe como excluyente por accidente.
    """
    if columnas_flag is None:
        columnas_flag = [c for c in df.columns if c.startswith("flag_")]
    columnas_flag = [c for c in columnas_flag
                      if not any(d in c for d in PRUEBAS_SOLO_DIAGNOSTICO)]
    if not columnas_flag:
        raise ValueError("No se encontraron columnas 'flag_*' evaluables (no-diagnosticas) "
                          "en el DataFrame. Corre primero aplicar_qc_fisico() y "
                          "aplicar_qc_experto().")

    es_flag = (df[columnas_flag] == "FLAG").any(axis=1)
    return pd.Series(np.where(es_flag, NO_USABLE, OK), index=df.index)


def aplicar_regla_islas(usabilidad: pd.Series, datetime_col: pd.Series,
                         ventana_minutos: int = 60) -> pd.Series:
    """
    Regla de islas: los datos "OK" que forman un segmento aislado entre
    tramos NO_USABLE cercanos (a menos de `ventana_minutos`, 60 por
    defecto) también se marcan NO_USABLE. Evita dejar islas sueltas de
    pocos puntos buenos en medio de un tramo malo, que no sirven para el
    análisis posterior.

    Implementación: para cada punto OK se busca el NO_USABLE más cercano
    hacia atrás y hacia adelante; si ambos están a <= ventana_minutos, el
    punto se descarta. Devuelve la serie de usabilidad actualizada.
    """
    u = usabilidad.copy()
    es_malo = (u == NO_USABLE)
    t = pd.DatetimeIndex(datetime_col)

    # Tiempo del NO_USABLE más cercano hacia atrás y hacia adelante
    serie_tiempo = pd.Series(t, index=u.index)
    tiempo_malo = serie_tiempo.where(es_malo)

    anterior = tiempo_malo.ffill()
    siguiente = tiempo_malo.bfill()

    ventana = pd.Timedelta(minutes=ventana_minutos)
    cerca_atras = (serie_tiempo - anterior) <= ventana
    cerca_adelante = (siguiente - serie_tiempo) <= ventana

    isla = (~es_malo) & anterior.notna() & siguiente.notna() & cerca_atras & cerca_adelante
    u.loc[isla] = NO_USABLE
    return u


def resumen_usabilidad(df: pd.DataFrame, col_usabilidad: str = "usabilidad") -> pd.DataFrame:
    """
    Tabla resumen: registros OK / NO_USABLE y % usable, en total y separado
    por día/noche (usa la columna 'elevation' de la geometría solar).
    """
    filas = []

    def _fila(etiqueta, subset):
        """Construye la fila de resumen (n, OK, NO_USABLE, % usable) de un subconjunto."""
        total = len(subset)
        if total == 0:
            return {"segmento": etiqueta, "n": 0, "OK": 0, "NO_USABLE": 0, "pct_usable": np.nan}
        ok = int((subset[col_usabilidad] == OK).sum())
        return {
            "segmento": etiqueta,
            "n": total,
            "OK": ok,
            "NO_USABLE": total - ok,
            "pct_usable": round(100 * ok / total, 3),
        }

    filas.append(_fila("TOTAL", df))
    if "elevation" in df.columns:
        filas.append(_fila("dia (elevation>0)", df[df["elevation"] > 0]))
        filas.append(_fila("noche (elevation<=0)", df[df["elevation"] <= 0]))

    return pd.DataFrame(filas)
