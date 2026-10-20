"""
verificar_pieza4_usabilidad.py
------------------------------
Paso 4 del pipeline: ejecuta geometria solar, QC fisico y QC experto y
combina todas las banderas en un veredicto de usabilidad por instante,
para cada fuente configurada (satelital y terrestre). Los nombres de
los scripts conservan una numeracion por paso del pipeline.

Se calculan dos reglas: la regla base (cualquier bandera FLAG implica
NO_USABLE) y la regla de islas del control de calidad experto, que ademas
marca como NO_USABLE los datos validos aislados entre tramos no usables
separados por menos de 60 min.

Entradas: config.yaml (umbrales y rutas) y los archivos crudos de
data/raw/ leidos mediante qcsolar.io.CARGADORES_TODAS.
Salidas:
  - data/processed/<fuente>_usabilidad.csv (banderas + columnas
    'usabilidad' y 'usabilidad_islas').
  - outputs/tablas/usabilidad_base_<fuente>.csv y
    outputs/tablas/usabilidad_islas_<fuente>.csv (total / dia / noche).
  - outputs/tablas/usabilidad_comparativo.csv (% usable por fuente).

Como ejecutarlo (con el entorno virtual activo):
    python verificar_pieza4_usabilidad.py
"""

import pandas as pd

from qcsolar.io import CARGADORES_TODAS
from qcsolar.solar import agregar_geometria_solar
from qcsolar.qc import physical as qcf
from qcsolar.qc import expert as qce
from qcsolar.qc import usability as qcu
from qcsolar import report

pd.set_option("display.width", 160)


def procesar(nombre, df):
    """Ejecuta geometria, QC fisico, QC experto y usabilidad (regla base y
    regla de islas) sobre una fuente, guarda sus tablas y dataset, y
    devuelve el % usable (ambas reglas) para la tabla comparativa final."""
    print(f"\n{'='*62}\n FUENTE: {nombre.upper()}\n{'='*62}")
    print("   Calculando geometria solar...")
    df = agregar_geometria_solar(df)
    print("   Aplicando QC fisico (Pieza 2)...")
    df = qcf.aplicar_qc_fisico(df, nombre)
    print("   Aplicando QC experto (Pieza 3)...")
    df = qce.aplicar_qc_experto(df, nombre)

    print("   Calculando usabilidad (regla base)...")
    df["usabilidad"] = qcu.calcular_usabilidad(df)

    print("   Calculando usabilidad (regla + islas <60 min)...")
    df["usabilidad_islas"] = qcu.aplicar_regla_islas(df["usabilidad"], df["datetime"])

    resumen_base = qcu.resumen_usabilidad(df, "usabilidad")
    resumen_islas = qcu.resumen_usabilidad(df, "usabilidad_islas")
    print("   --- Usabilidad (regla base) ---")
    print(resumen_base.to_string(index=False))
    print("   --- Usabilidad (regla + islas) ---")
    print(resumen_islas.to_string(index=False))

    report.guardar_tabla(resumen_base, f"usabilidad_base_{nombre}.csv")
    report.guardar_tabla(resumen_islas, f"usabilidad_islas_{nombre}.csv")
    report.guardar_dataset(df, f"{nombre}_usabilidad.csv", etapa="processed")

    fila_total_base = resumen_base[resumen_base["segmento"] == "TOTAL"].iloc[0]
    fila_total_islas = resumen_islas[resumen_islas["segmento"] == "TOTAL"].iloc[0]
    return {
        "fuente": nombre,
        "pct_usable_base": fila_total_base["pct_usable"],
        "pct_usable_islas": fila_total_islas["pct_usable"],
    }


def main():
    """Calcula la usabilidad de cada fuente y guarda la tabla comparativa de % usable."""
    print("=" * 62)
    print(" VERIFICACION PIEZA 4 - Usabilidad (2 fuentes)")
    print("=" * 62)

    filas_comparativo = []
    for nombre, cargador in CARGADORES_TODAS.items():
        df = cargador()
        filas_comparativo.append(procesar(nombre, df))

    comparativo = pd.DataFrame(filas_comparativo)
    print(f"\n{'='*62}\n TABLA COMPARATIVA - %% USABLE por fuente\n{'='*62}")
    print(comparativo.to_string(index=False))
    report.guardar_tabla(comparativo, "usabilidad_comparativo.csv")

    print("\n" + "=" * 62)
    print(" PIEZA 4 LISTA. Usabilidad calculada para las 2 fuentes.")
    print("   Revisa outputs/tablas/usabilidad_comparativo.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()
