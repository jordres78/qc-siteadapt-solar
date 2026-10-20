"""
preparar_udenar_ghi.py
-----------------------
Convierte la potencia AC del inversor de la estacion terrestre en
irradiancia global horizontal (GHI) estimada:

    GHI [W/m2] = P [W] / (eta * N * A)

con eta = 0.2133 (eficiencia del modulo), N = 30 (numero de modulos) y
A = 2.56 m2 (area de cada modulo). Se convierte el archivo completo, con
el mismo muestreo de 5 min de la serie original.

No se recorta ni se filtra ningun valor (negativos, nocturnos, etc.):
esa tarea corresponde al QC fisico y experto de los pasos siguientes.
Aqui solo se realiza la conversion de unidades.

Entrada: RUTA_ENTRADA (data/raw/y.csv, columnas 'date' y 'potencia').
Salida: RUTA_SALIDA (CSV en data/raw/ con columnas 'date' y 'ghi'), en
el formato que espera el cargador de la estacion en qcsolar.io.

Como ejecutarlo (con el entorno virtual activo, desde la raiz del
proyecto):
    python preparar_udenar_ghi.py
"""

import pandas as pd
from pathlib import Path

RUTA_ENTRADA = Path("data/raw/y.csv")
RUTA_SALIDA = Path("data/raw/udenar.csv")

# Denominador eta * N * A de la conversion potencia -> irradiancia:
# GHI (W/m2) = potencia (W) / (0.2133 * 30 * 2.56)
FACTOR_CONVERSION = 0.2133 * 30 * 2.56


def main():
    """Lee la potencia del inversor, la convierte a GHI y guarda el CSV resultante."""
    print(f"Leyendo {RUTA_ENTRADA} ...")
    df = pd.read_csv(RUTA_ENTRADA, parse_dates=["date"])
    print(f"   {len(df):,} registros leidos.")
    print(f"   Rango de fechas: {df['date'].min()} -> {df['date'].max()}")

    df["ghi"] = df["potencia"] / FACTOR_CONVERSION
    salida = df[["date", "ghi"]].copy()

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(RUTA_SALIDA, index=False)
    print(f"Guardado {RUTA_SALIDA} ({len(salida):,} registros, columnas: date, ghi).")
    print("Listo -- este archivo ya lo puede leer qcsolar.io.cargar_udenar().")


if __name__ == "__main__":
    main()
