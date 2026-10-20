"""
verificar_pieza3_qc_experto.py
------------------------------
Paso 3 del pipeline: control de calidad experto sobre cada fuente
configurada (satelital y terrestre). La fuente satelital es la unica
con DNI/DHI, por lo que solo en ella se evaluan el cierre radiativo y
las pruebas K; en la estacion terrestre se evaluan indice de cielo
claro (CSI), tasa de cambio, persistencia y simetria AM/PM. Los nombres
de los scripts conservan una numeracion por paso del pipeline.

Antes del QC experto se aplican la geometria solar y el QC fisico, de
modo que todas las banderas quedan en un mismo dataset.

Entradas: config.yaml (umbrales y rutas) y los archivos crudos de
data/raw/ leidos mediante qcsolar.io.CARGADORES_TODAS.
Salidas:
  - outputs/tablas/qc_experto_flags_<fuente>.csv y
    outputs/tablas/qc_experto_flags_comparativo.csv.
  - data/processed/<fuente>_qc_completo.csv (todas las banderas).
  - outputs/figuras/: cierre radiativo (satelital) y CSI diario.

Como ejecutarlo (con el entorno virtual activo):
    python verificar_pieza3_qc_experto.py
"""

import pandas as pd

from qcsolar.io import CARGADORES_TODAS
from qcsolar.solar import agregar_geometria_solar
from qcsolar.qc import physical as qcf
from qcsolar.qc import expert as qce
from qcsolar import report, viz

pd.set_option("display.width", 160)


def procesar(nombre, df):
    """Aplica geometria solar, QC fisico y QC experto a una fuente, guarda su
    dataset y tabla de banderas, genera sus figuras y devuelve la columna de
    % de banderas de esa fuente (para la tabla comparativa)."""
    print(f"\n{'='*62}\n FUENTE: {nombre.upper()}\n{'='*62}")
    print("   Calculando geometria solar...")
    df = agregar_geometria_solar(df)
    print("   Aplicando QC fisico (Pieza 2)...")
    df = qcf.aplicar_qc_fisico(df, nombre)
    print("   Aplicando QC experto (Pieza 3)...")
    df = qce.aplicar_qc_experto(df, nombre)

    resumen = qce.resumen_flags_experto(df)
    print("   --- Resumen flags experto ---")
    print(resumen.to_string(index=False))
    report.guardar_tabla(resumen, f"qc_experto_flags_{nombre}.csv")
    report.guardar_dataset(df, f"{nombre}_qc_completo.csv", etapa="processed")

    color = viz.COLOR_SOLCAST if nombre.startswith("solcast") else viz.COLOR_ALVERNIA

    if nombre.startswith("solcast"):
        # Solo la fuente satelital dispone de las 3 componentes para el cierre radiativo
        viz.fig_cierre_radiativo(
            df, f"{nombre.upper()} - cierre radiativo (GHI medido vs calculado)",
            f"13_{nombre}_cierre_radiativo.png")

    viz.fig_csi_anual(
        df, f"{nombre.upper()} - indice de cielo claro (CSI) diario",
        f"14_{nombre}_csi_anual.png", color=color)

    return resumen.set_index("prueba")["pct_flag"].rename(nombre)


def main():
    """Aplica el QC experto a cada fuente y guarda la tabla comparativa de banderas."""
    print("=" * 62)
    print(" VERIFICACION PIEZA 3 - QC experto (2 fuentes: satelital + terrestre)")
    print("=" * 62)

    columnas_comparativas = []
    for nombre, cargador in CARGADORES_TODAS.items():
        df = cargador()
        columnas_comparativas.append(procesar(nombre, df))

    comparativo = pd.concat(columnas_comparativas, axis=1).reset_index()
    comparativo = comparativo.rename(columns={"index": "prueba"})
    print(f"\n{'='*62}\n TABLA COMPARATIVA - %% FLAG por prueba y fuente (2 fuentes)\n{'='*62}")
    print(comparativo.to_string(index=False))
    report.guardar_tabla(comparativo, "qc_experto_flags_comparativo.csv")

    print("\n" + "=" * 62)
    print(" PIEZA 3 LISTA. QC experto aplicado a las 2 fuentes.")
    print("   Revisa outputs/tablas/qc_experto_flags_comparativo.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()
