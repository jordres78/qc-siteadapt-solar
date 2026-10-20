"""
diagnostico_violaciones_fisicas.py
-------------------------------------------------------------------------------
Herramienta de diagnostico que inspecciona, instante por instante, las
violaciones de las pruebas fisicas BSRN que producen los modelos de
MODELOS_A_INSPECCIONAR al aplicarse sobre el periodo completo de
aplicacion. Estas violaciones son las que hacen que el criterio de
seleccion descarte un modelo aunque tenga menor RMSE en test.

Para cada estacion de site_adaptation.reconstruccion.fuentes_terrestres
ejecuta los pasos 1-4, el emparejamiento y el entrenamiento, y luego
aplica el QC fisico a las predicciones de cada modelo inspeccionado.

Entradas: config.yaml y los archivos crudos de data/raw/.
Salidas: ninguna en disco; imprime el detalle de cada instante invalido
(fecha, prediccion, elevacion solar y prueba disparada) para distinguir
violaciones marginales cercanas al limite de errores de extrapolacion.

Como ejecutarlo (con el entorno virtual activo):
    python diagnostico_violaciones_fisicas.py
"""

import pandas as pd

from qcsolar.io import cargar_solcast, CARGADORES_TERRESTRES
from qcsolar.solar import agregar_geometria_solar
from qcsolar.qc import physical as qcf
from qcsolar.qc import expert as qce
from qcsolar.qc import usability as qcu
from qcsolar.siteadapt import prepare as prep
from qcsolar.siteadapt import evaluate as ev
from qcsolar.config import CONFIG

pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.3f}".format)

RECON_CFG = CONFIG["site_adaptation"]["reconstruccion"]
FUENTES_TERRESTRES = RECON_CFG["fuentes_terrestres"]
EXCLUSIONES = RECON_CFG.get("exclusiones", {})
MINIMO_FISICO = CONFIG["qc_fisico"]["minimo_irradiancia"]
FREQ = RECON_CFG.get("resolucion", "5min")

MODELOS_A_INSPECCIONAR = ["Extra Trees", "Random Forest"]

COLUMNAS_PREDICTORAS = [
    "GHI_solcast", "DNI_solcast", "DHI_solcast", "GTI_solcast",
    "TEMP_solcast", "RH_solcast", "WS_solcast", "CLOUD_OPACITY_solcast",
    "zenith_solcast", "elevation_solcast", "hour", "dayofyear", "month",
]


def pipeline_completo(nombre, df):
    """Ejecuta los pasos 1-4 del pipeline (geometria solar, QC fisico, QC
    experto, usabilidad) sobre una fuente cruda."""
    df =agregar_geometria_solar(df)
    df = qcf.aplicar_qc_fisico(df, nombre)
    df = qce.aplicar_qc_experto(df, nombre)
    df["usabilidad"] = qcu.calcular_usabilidad(df)
    return df


def main():
    """Entrena los modelos por estacion e imprime los instantes en que las
    predicciones de los modelos inspeccionados violan las pruebas fisicas."""
    print("Cargando y procesando fuentes...")
    sol = pipeline_completo("solcast", cargar_solcast())

    for clave in FUENTES_TERRESTRES:
        if clave not in CARGADORES_TERRESTRES:
            continue
        est = pipeline_completo(clave, CARGADORES_TERRESTRES[clave]())
        excl = EXCLUSIONES.get(clave, {}) or {}
        supervisado, aplicacion = prep.emparejar_horario_generico(
            sol, est, nombre_objetivo=clave,
            excluir_desde=excl.get("desde"), excluir_hasta=excl.get("hasta"),
            freq=FREQ,
        )
        train, test = prep.split_temporal(supervisado)
        target = f"GHI_{clave}"
        columnas_disp = [c for c in COLUMNAS_PREDICTORAS if c in supervisado.columns]

        print(f"Entrenando los 9 modelos para {clave}...")
        _, _, modelos_entrenados = ev.entrenar_y_evaluar(
            train, test, target=target, columnas_predictoras=columnas_disp)

        aplicacion_geo = agregar_geometria_solar(aplicacion.reset_index(drop=True), "datetime")

        for nombre_modelo in MODELOS_A_INSPECCIONAR:
            if nombre_modelo not in modelos_entrenados:
                continue
            modelo = modelos_entrenados[nombre_modelo]
            pred = modelo.predict(aplicacion)

            tmp = aplicacion_geo[["datetime", "zenith", "elevation", "dni_extra"]].copy()
            tmp["GHI"] = pred
            tmp = qcf.aplicar_qc_fisico(tmp, "terrestre")
            cols_flag = [c for c in tmp.columns if c.startswith("flag_")]
            es_invalido = (tmp[cols_flag] == "FLAG").any(axis=1)

            print(f"\n{'=' * 78}")
            print(f" {clave.upper()} -- {nombre_modelo}: {int(es_invalido.sum())} instantes invalidos de {len(tmp):,}")
            print(f"{'=' * 78}")
            if es_invalido.sum() == 0:
                print("   (ninguno)")
                continue

            detalle = tmp.loc[es_invalido, ["datetime", "GHI", "zenith", "elevation", "dni_extra"] + cols_flag].copy()
            for c in cols_flag:
                detalle[c] = detalle[c].where(detalle[c] == "FLAG", "")
            print(detalle.to_string(index=False))

    print("\nListo. Revisa arriba: si el valor de GHI en esos instantes esta apenas "
          "fuera del limite (ej. -4.x W/m2 con el minimo en -4.0, o un poco por "
          "encima del limite PPL/ERL al amanecer/atardecer), es un borde trivial. "
          "Si son valores muy alejados del limite, es una extrapolacion real.")


if __name__ == "__main__":
    main()
