"""
metrics.py
----------
Metricas de regresion estandar para evaluar el site-adaptation: MAE,
RMSE, MBE, rMBE, rRMSE, coeficiente de Pearson y R2. Deben calcularse
siempre sobre un mismo conjunto de datos (sin mezclar train y test).
Las metricas relativas se expresan en % de la media de la referencia.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metricas_regresion(y_real, y_pred) -> dict:
    """
    Calcula el conjunto completo de metricas comparando y_real vs y_pred.
    Descarta las filas donde alguno de los dos sea NaN; con menos de 2
    pares validos devuelve todas las metricas en NaN. MBE = media(pred-real).
    """
    y_real = pd.Series(y_real).astype(float).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(float).reset_index(drop=True)
    validos = pd.concat([y_real, y_pred], axis=1).dropna()
    validos.columns = ["y_real", "y_pred"]

    if len(validos) < 2:
        return {"MAE": np.nan, "RMSE": np.nan, "MBE": np.nan, "rMBE": np.nan,
                "rRMSE": np.nan, "Pearson_R": np.nan, "R2": np.nan, "n": len(validos)}

    mae = mean_absolute_error(validos["y_real"], validos["y_pred"])
    rmse = float(np.sqrt(mean_squared_error(validos["y_real"], validos["y_pred"])))
    mbe = (validos["y_pred"] - validos["y_real"]).mean()
    media_ref = validos["y_real"].mean()

    pearson = validos["y_real"].corr(validos["y_pred"]) if validos["y_real"].std() > 0 and validos["y_pred"].std() > 0 else np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MBE": mbe,
        "rMBE": 100 * mbe / media_ref if media_ref != 0 else np.nan,
        "rRMSE": 100 * rmse / media_ref if media_ref != 0 else np.nan,
        "Pearson_R": pearson,
        "R2": r2_score(validos["y_real"], validos["y_pred"]),
        "n": len(validos),
    }


def porcentaje_mejora(antes: float, despues: float, mas_alto_es_mejor: bool = False) -> float:
    """
    % de mejora de una metrica entre "antes" (fuente satelital cruda) y
    "despues" (modelo adaptado), relativo a |antes|. Para MAE/RMSE/rRMSE
    (mas bajo = mejor) usar mas_alto_es_mejor=False (por defecto); para
    R2/Pearson (mas alto = mejor), mas_alto_es_mejor=True. NaN si antes=0.
    """
    if pd.isna(antes) or pd.isna(despues) or antes == 0:
        return np.nan
    if mas_alto_es_mejor:
        return 100 * (despues - antes) / abs(antes)
    return 100 * (antes - despues) / abs(antes)
