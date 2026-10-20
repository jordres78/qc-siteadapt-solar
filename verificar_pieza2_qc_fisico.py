"""
verificar_pieza2_qc_fisico.py
-----------------------------
Paso 2 del pipeline: control de calidad fisico sobre cada fuente
configurada (satelital y terrestre). Aplica las pruebas de valor
negativo, nocturno, limite fisicamente posible (PPL, BSRN), limite
extremadamente raro (ERL) y sensor caido. Los nombres de los scripts
conservan una numeracion por paso del pipeline.

Entradas: config.yaml (umbrales y rutas) y los archivos crudos de
data/raw/ leidos mediante qcsolar.io.CARGADORES_TODAS.
Salidas:
  - outputs/tablas/qc_fisico_flags_<fuente>.csv (conteo de banderas).
  - outputs/tablas/qc_fisico_flags_comparativo.csv (% de banderas por
    prueba y fuente).
  - data/processed/<fuente>_qc_fisico.csv (dataset con banderas).
  - outputs/figuras/: datos marcados y banderas por mes, por fuente.

Como ejecutarlo (con el entorno virtual activo):
    python verificar_pieza2_qc_fisico.py
"""

import pandas as pd

from qcsolar.io import CARGADORES_TODAS
from qcsolar.solar import agregar_geometria_solar
from qcsolar.qc import physical as qcf
from qcsolar import report, viz

pd.set_option("display.width", 160)


def procesar(nombre, df):
    """Aplica geometria solar y QC fisico a una fuente, guarda su dataset y
    tabla de banderas, genera sus 2 figuras de diagnostico y devuelve la
    columna de % de banderas de esa fuente (para la tabla comparativa)."""
    print(f"\n{'='*60}\n FUENTE: {nombre.upper()}\n{'='*60}")
    print("   Calculando geometria solar...")
    df = agregar_geometria_solar(df)
    print("   Aplicando QC fisico...")
    df = qcf.aplicar_qc_fisico(df, nombre)

    resumen = qcf.resumen_flags(df)
    print("   --- Resumen de flags ---")
    print(resumen.to_string(index=False))
    report.guardar_tabla(resumen, f"qc_fisico_flags_{nombre}.csv")
    report.guardar_dataset(df, f"{nombre}_qc_fisico.csv", etapa="processed")

    color = viz.COLOR_SOLCAST if nombre.startswith("solcast") else viz.COLOR_ALVERNIA
    viz.fig_datos_flagueados(
        df, "GHI", "flag_ppl_GHI",
        f"{nombre.upper()} - GHI: datos que superan el limite fisico BSRN",
        f"11_{nombre}_ghi_flagueado.png", fuente_color=color)
    viz.fig_flags_por_mes(
        df, "flag_ppl_GHI",
        f"{nombre.upper()} - datos marcados por mes (limite fisico GHI)",
        f"12_{nombre}_flags_por_mes.png")

    return resumen.set_index("prueba")["pct_flag"].rename(nombre)


def main():
    """Aplica el QC fisico a cada fuente y guarda la tabla comparativa de banderas."""
    print("=" * 60)
    print(" VERIFICACION PIEZA 2 - QC fisico (2 fuentes: satelital + terrestre)")
    print("=" * 60)

    columnas_comparativas = []
    for nombre, cargador in CARGADORES_TODAS.items():
        df = cargador()
        columnas_comparativas.append(procesar(nombre, df))

    comparativo = pd.concat(columnas_comparativas, axis=1).reset_index()
    comparativo = comparativo.rename(columns={"index": "prueba"})
    print(f"\n{'='*60}\n TABLA COMPARATIVA - %% FLAG por prueba y fuente (2 fuentes)\n{'='*60}")
    print(comparativo.to_string(index=False))
    report.guardar_tabla(comparativo, "qc_fisico_flags_comparativo.csv")

    print("\n" + "=" * 60)
    print(" PIEZA 2 LISTA. QC fisico aplicado a las 2 fuentes.")
    print("   Revisa outputs/tablas/qc_fisico_flags_comparativo.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
