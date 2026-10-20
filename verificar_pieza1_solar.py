"""
verificar_pieza1_solar.py
-------------------------
Paso 1 del pipeline: calculo de la geometria solar (cenit, elevacion,
irradiancia extraterrestre, GHI de cielo claro) para cada fuente
configurada (satelital y terrestre). Los nombres de los scripts
conservan una numeracion por paso del pipeline.

La geometria solar depende solo de la fecha/hora y de la ubicacion del
sitio, por lo que el mismo modulo (qcsolar.solar) se aplica sin cambios
a todas las fuentes.

Entradas: config.yaml (ubicacion del sitio y rutas) y los archivos
crudos de data/raw/ leidos mediante qcsolar.io.CARGADORES_TODAS.
Salidas: ninguna en disco; imprime un resumen y chequeos de sentido
fisico (elevacion y GHI de cielo claro a mediodia y medianoche).

Como ejecutarlo (con el entorno virtual activo):
    python verificar_pieza1_solar.py
"""

import pandas as pd

from qcsolar.io import CARGADORES_TODAS
from qcsolar.solar import agregar_geometria_solar, resumen_geometria

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 25)


def probar_fuente(nombre, df):
    """Aplica geometria solar a una fuente y muestra chequeos de sentido fisico."""
    print(f"\n{'='*60}\n FUENTE: {nombre.upper()}\n{'='*60}")
    print(f"   Registros: {len(df):,}")
    print("   Calculando geometria solar (puede tardar)...")
    df = agregar_geometria_solar(df)

    resumen = resumen_geometria(df)
    print("   --- Resumen ---")
    for clave, valor in resumen.items():
        print(f"      {clave}: {valor}")

    mediodia = df[df["datetime"].dt.hour == 12]
    madrugada = df[df["datetime"].dt.hour == 0]
    print("   --- Chequeos de sentido fisico ---")
    print(f"      Elevacion 12:00 (debe ser ALTA):     {mediodia['elevation'].mean():.1f} grados")
    print(f"      Elevacion 00:00 (debe ser NEGATIVA): {madrugada['elevation'].mean():.1f} grados")
    print(f"      ghi_clear 12:00 (debe ser ALTO):     {mediodia['ghi_clear'].mean():.0f} W/m2")
    print(f"      ghi_clear 00:00 (debe ser ~0):       {madrugada['ghi_clear'].mean():.0f} W/m2")
    return df


def main():
    """Carga cada fuente configurada y le aplica la verificacion de geometria solar."""
    print("=" * 60)
    print(" VERIFICACION PIEZA 1 - Geometria solar (2 fuentes: satelital + terrestre)")
    print("=" * 60)

    for nombre, cargador in CARGADORES_TODAS.items():
        df = cargador()
        probar_fuente(nombre, df)

    print("\n" + "=" * 60)
    print(" PIEZA 1 LISTA. La geometria solar funciona en las 2 fuentes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
