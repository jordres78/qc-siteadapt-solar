"""
verificar_pieza5b_emparejamiento_4estaciones.py
-----------------------------------------------
Paso 5 del pipeline: emparejamiento de la serie satelital con la serie
de la estacion terrestre, a la resolucion definida en config.yaml (5 min
por defecto), y division temporal train/test (70/30). Los nombres de
los scripts conservan una numeracion por paso del pipeline.

Se procesa cada estacion terrestre listada en
site_adaptation.reconstruccion.fuentes_terrestres; para cada una se usa
como referencia (variables predictoras) la fuente satelital asociada en
satelital_por_fuente. No se fija una ventana de fechas: se filtra
instante por instante con la bandera de usabilidad y solo se excluyen
las ventanas indicadas en site_adaptation.reconstruccion.exclusiones.

Por cada estacion se generan dos conjuntos
(prep.emparejar_horario_generico):
  - supervisado: instantes con dato terrestre usable, para entrenar y
                 evaluar modelos.
  - aplicacion:  todos los instantes con dato satelital, para reconstruir
                 la serie completa con el modelo seleccionado.

Este script no entrena modelos; solo prepara los datos.

Entradas: config.yaml y los archivos crudos de data/raw/ (fuente
satelital y estaciones terrestres).
Salidas:
  - data/processed/reconstruccion_<estacion>_{supervisado,aplicacion,
    train,test}.csv.
  - outputs/tablas/reconstruccion_resumen_emparejamiento.csv.

Como ejecutarlo (con el entorno virtual activo):
    python verificar_pieza5b_emparejamiento_4estaciones.py
"""

import pandas as pd

from qcsolar.io import cargar_solcast, CARGADORES_TERRESTRES
from qcsolar.solar import agregar_geometria_solar
from qcsolar.qc import physical as qcf
from qcsolar.qc import expert as qce
from qcsolar.qc import usability as qcu
from qcsolar.siteadapt import prepare as prep
from qcsolar.config import CONFIG
from qcsolar import report

pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.3f}".format)

RECON_CFG = CONFIG["site_adaptation"]["reconstruccion"]
FUENTES_TERRESTRES = RECON_CFG["fuentes_terrestres"]
EXCLUSIONES = RECON_CFG.get("exclusiones", {})
# Resolucion de trabajo leida de config.yaml (5 min nativos por defecto). Se
# pasa explicitamente porque el valor por defecto de
# emparejar_horario_generico es "1h"; asi el emparejamiento, el entrenamiento
# y la reconstruccion usan la misma resolucion y los conteos coinciden.
FREQ = RECON_CFG.get("resolucion", "5min")


def pipeline_completo(nombre, df):
    """Ejecuta los pasos 1-4 del pipeline (geometria solar, QC fisico, QC
    experto, usabilidad) sobre una fuente cruda, en el orden requerido."""
    print(f"   [{nombre}] geometria solar...")
    df = agregar_geometria_solar(df)
    print(f"   [{nombre}] QC fisico...")
    df = qcf.aplicar_qc_fisico(df, nombre)
    print(f"   [{nombre}] QC experto...")
    df = qce.aplicar_qc_experto(df, nombre)
    print(f"   [{nombre}] usabilidad...")
    df["usabilidad"] = qcu.calcular_usabilidad(df)
    return df


def main():
    """Empareja cada estacion terrestre con su serie satelital, divide en
    train/test y guarda los conjuntos y la tabla resumen del emparejamiento."""
    print("=" * 70)
    print(" PIEZA 5b - Emparejamiento fuente satelital -> fuente(s) terrestre(s)")
    print("=" * 70)

    # Cada estacion terrestre tiene su propia serie satelital asociada
    # (config.yaml: satelital_por_fuente), que debe solaparse en el tiempo
    # con ella; de lo contrario el conjunto supervisado queda vacio. La serie
    # satelital procesada se guarda en cache por si varias estaciones
    # comparten la misma.
    _cache_solcast = {}

    def _solcast_para(clave_terrestre):
        """Devuelve la serie satelital asociada a la estacion, ya procesada
        con los pasos 1-4 (se calcula una sola vez por serie satelital)."""
        clave_sat = CONFIG.get("satelital_por_fuente", {}).get(clave_terrestre, "solcast_raw")
        if clave_sat not in _cache_solcast:
            print(f"\n Procesando fuente SATELITAL ({clave_sat}, pareja de '{clave_terrestre}')...")
            _cache_solcast[clave_sat] = pipeline_completo(clave_sat, cargar_solcast(clave_terrestre))
        return _cache_solcast[clave_sat]

    resumenes = []
    for clave in FUENTES_TERRESTRES:
        if clave not in CARGADORES_TERRESTRES:
            print(f"\n AVISO: '{clave}' no esta en CARGADORES_TERRESTRES, se omite.")
            continue

        print(f"\n{'-' * 70}\n Procesando {clave.upper()}...")
        sol = _solcast_para(clave)
        est = pipeline_completo(clave, CARGADORES_TERRESTRES[clave]())

        excl = EXCLUSIONES.get(clave, {}) or {}
        excluir_desde = excl.get("desde")
        excluir_hasta = excl.get("hasta")
        if excluir_desde:
            print(f"   Exclusion de entrenamiento: desde {excluir_desde}"
                  f"{' hasta ' + excluir_hasta if excluir_hasta else ' (sin fin)'}")

        supervisado, aplicacion = prep.emparejar_horario_generico(
            sol, est, nombre_objetivo=clave,
            periodo_inicio=None, periodo_fin=None,   # sin ventana fija: filtro por instante usable
            excluir_desde=excluir_desde, excluir_hasta=excluir_hasta,
            freq=FREQ,
        )
        print(f"   Muestras supervisadas ({FREQ}, train+test): {len(supervisado):,}")
        print(f"   Muestras de aplicacion ({FREQ}, reconstruccion completa): {len(aplicacion):,}")
        print(f"   Distribucion origen_dato (aplicacion): "
              f"{aplicacion['origen_dato'].value_counts().to_dict()}")

        if supervisado.empty:
            print(f"   AVISO: 0 horas supervisadas para {clave}. No se puede entrenar. Se omite del split.")
            continue

        train, test = prep.split_temporal(supervisado)
        resumen = prep.resumen_emparejamiento_generico(supervisado, aplicacion, clave, train, test)
        print("   --- Resumen ---")
        for k, v in resumen.items():
            print(f"      {k}: {v}")
        resumenes.append(resumen)

        report.guardar_dataset(supervisado, f"reconstruccion_{clave}_supervisado.csv", etapa="processed")
        report.guardar_dataset(aplicacion, f"reconstruccion_{clave}_aplicacion.csv", etapa="processed")
        report.guardar_dataset(train, f"reconstruccion_{clave}_train.csv", etapa="processed")
        report.guardar_dataset(test, f"reconstruccion_{clave}_test.csv", etapa="processed")

    if resumenes:
        report.guardar_tabla(pd.DataFrame(resumenes), "reconstruccion_resumen_emparejamiento.csv")

    print("\n" + "=" * 70)
    print(" PIEZA 5b LISTA. Datos preparados para entrenar modelos (Pieza 6c/7b).")
    print("=" * 70)


if __name__ == "__main__":
    main()
