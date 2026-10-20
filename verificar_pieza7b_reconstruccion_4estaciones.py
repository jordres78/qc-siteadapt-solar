"""
verificar_pieza7b_reconstruccion_4estaciones.py
-----------------------------------------------
Paso 7 del pipeline: reconstruccion de la serie de GHI de la estacion
terrestre y reaplicacion del QC. Los nombres de los scripts conservan
una numeracion por paso del pipeline.

Para cada estacion de site_adaptation.reconstruccion.fuentes_terrestres
el script es autocontenido (no depende de CSV de pasos anteriores):
  1. Ejecuta los pasos 1-4, el emparejamiento y la division temporal
     70/30.
  2. Entrena todos los modelos (nueve) y selecciona uno con el mismo
     criterio de verificar_pieza6c_modelos_4estaciones.py: se descarta
     todo modelo cuyas predicciones sobre el periodo completo de
     aplicacion violen las pruebas fisicas BSRN (negativo, PPL, ERL,
     nocturno; tolerancia nocturna de 0,01 %), y entre los validos gana
     el de menor RMSE en test, sin exclusion por familia de modelo.
  3. Construye dos variantes de la serie corregida sobre todo el periodo
     disponible de la fuente satelital:
       - GHI_corregido_total:     prediccion del modelo en todos los
                                  instantes.
       - GHI_corregido_selectivo: conserva el dato medido que pasa el QC,
                                  corrige lo marcado y rellena huecos.
  4. Reaplica el mismo QC fisico y experto a las tres series (cruda,
     corregida total, corregida selectiva) y compara % de banderas y
     % usable.

Resolucion: se lee de config.yaml (site_adaptation.reconstruccion.resolucion,
5 min nativos por defecto). Se trabaja a resolucion nativa porque un
modelo puede ajustar bien en promedios horarios y aun asi producir
valores fisicamente imposibles a la resolucion de los datos.

Validez fisica: se evalua sobre el periodo completo de aplicacion y no
solo sobre el 30 % de prueba, porque un modelo sin violaciones en test
puede producirlas en el resto del periodo (ver
evaluate.evaluar_consistencia_fisica_aplicacion()).

Entradas: config.yaml y los archivos crudos de data/raw/.
Salidas:
  - data/processed/<estacion>_ghi_corregido.csv (serie original, dos
    variantes corregidas, origen del dato y usabilidad de cada serie).
  - outputs/tablas/reconstruccion_<estacion>_flags_comparativo.csv.
  - outputs/tablas/reconstruccion_metricas_modelos.csv.
  - outputs/tablas/reconstruccion_comparativo_4estaciones.csv (% usable
    por estacion y variante).

Como ejecutarlo (con el entorno virtual activo; puede tardar varios
minutos, pues entrena todos los modelos y aplica el chequeo fisico al
periodo completo):
    python verificar_pieza7b_reconstruccion_4estaciones.py
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
from qcsolar import report

pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.3f}".format)

RECON_CFG = CONFIG["site_adaptation"]["reconstruccion"]
FUENTES_TERRESTRES = RECON_CFG["fuentes_terrestres"]
EXCLUSIONES = RECON_CFG.get("exclusiones", {})
MINIMO_FISICO = CONFIG["qc_fisico"]["minimo_irradiancia"]
FREQ = RECON_CFG.get("resolucion", "5min")

COLUMNAS_PREDICTORAS = [
    "GHI_solcast", "DNI_solcast", "DHI_solcast", "GTI_solcast",
    "TEMP_solcast", "RH_solcast", "WS_solcast", "CLOUD_OPACITY_solcast",
    "zenith_solcast", "elevation_solcast", "hour", "dayofyear", "month",
]


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
    """Para cada estacion selecciona el modelo, construye las dos variantes
    corregidas, reaplica el QC y guarda series y tablas comparativas."""
    print("=" * 72)
    print(f" PIEZA 7b - Reconstruccion (2 variantes) + re-QC, resolucion {FREQ}")
    print("=" * 72)

    # Cada estacion terrestre tiene su propia serie satelital asociada
    # (config.yaml: satelital_por_fuente). La serie satelital procesada se
    # guarda en cache para no reprocesarla si varias estaciones la comparten.
    _cache_solcast = {}

    def _solcast_para(clave_terrestre):
        """Devuelve la serie satelital asociada a la estacion, ya procesada
        con los pasos 1-4 (se calcula una sola vez por serie satelital)."""
        clave_sat = CONFIG.get("satelital_por_fuente", {}).get(clave_terrestre, "solcast_raw")
        if clave_sat not in _cache_solcast:
            print(f"\n Procesando fuente SATELITAL ({clave_sat}, pareja de '{clave_terrestre}')...")
            _cache_solcast[clave_sat] = pipeline_completo(clave_sat, cargar_solcast(clave_terrestre))
        return _cache_solcast[clave_sat]

    filas_comparativa = []
    todas_metricas = []
    for clave in FUENTES_TERRESTRES:
        if clave not in CARGADORES_TERRESTRES:
            continue
        print(f"\n{'=' * 72}\n ESTACION: {clave.upper()}  (resolucion {FREQ})\n{'=' * 72}")
        sol = _solcast_para(clave)
        est = pipeline_completo(clave, CARGADORES_TERRESTRES[clave]())

        excl = EXCLUSIONES.get(clave, {}) or {}
        supervisado, aplicacion = prep.emparejar_horario_generico(
            sol, est, nombre_objetivo=clave,
            excluir_desde=excl.get("desde"), excluir_hasta=excl.get("hasta"),
            freq=FREQ,
        )
        if supervisado.empty:
            print(f"   AVISO: 0 muestras supervisadas para {clave}. Se omite esta estacion.")
            continue

        train, test = prep.split_temporal(supervisado)
        target = f"GHI_{clave}"
        columnas_disp = [c for c in COLUMNAS_PREDICTORAS if c in supervisado.columns]

        print(f"   Entrenando los 9 modelos (train={len(train):,}, test={len(test):,})...")
        tabla_metricas, predicciones_test, modelos_entrenados = ev.entrenar_y_evaluar(
            train, test, target=target, columnas_predictoras=columnas_disp)

        # --- Filtro de validez fisica sobre el periodo completo de
        #     aplicacion, no solo el 30% de prueba (ver docstring del modulo) ---
        print(f"   Aplicando los 9 modelos sobre TODO el periodo ({len(aplicacion):,} muestras de {FREQ})"
              " para el chequeo de validez fisica...")
        aplicacion_geo = agregar_geometria_solar(aplicacion.reset_index(drop=True), "datetime")
        predicciones_aplicacion_todos = ev.predecir_aplicacion_todos_modelos(aplicacion, modelos_entrenados)

        ganador, mejor_rmse_puro, tabla_metricas = ev.elegir_mejor_modelo(
            tabla_metricas, predicciones_test, target, MINIMO_FISICO,
            predicciones_aplicacion=predicciones_aplicacion_todos,
            aplicacion_con_geometria=aplicacion_geo,
        )
        rmse_ganador = tabla_metricas.loc[tabla_metricas["modelo"] == ganador, "RMSE"].iloc[0]
        pct_inval_ganador = tabla_metricas.loc[tabla_metricas["modelo"] == ganador, "pct_fisicamente_invalido"].iloc[0]
        print(f"   Modelo elegido: {ganador}  (RMSE test = {rmse_ganador:.2f} W/m2, a {FREQ}; "
              f"{pct_inval_ganador:.3f}% fisicamente invalido en TODO el periodo de aplicacion)")

        modelo_ganador = modelos_entrenados[ganador]

        print(f"   Construyendo las 2 variantes con el modelo ganador...")
        reconstruida = ev.construir_serie_corregida(aplicacion, modelo_ganador, clave)

        # Geometria SIN sufijo, recalculada sobre el datetime de la reconstruccion
        # (necesaria para que qc.physical/qc.expert puedan evaluar las 3 series).
        reconstruida = agregar_geometria_solar(reconstruida.reset_index(drop=True), "datetime")

        col_crudo = f"GHI_{clave}_crudo"
        datos = reconstruida
        datos, cols_crudo, usab_crudo = ev.aplicar_qc_a_serie(datos, col_crudo, "crudo")
        datos, cols_total, usab_total = ev.aplicar_qc_a_serie(datos, "GHI_corregido_total", "total")
        datos, cols_selectivo, usab_selectivo = ev.aplicar_qc_a_serie(datos, "GHI_corregido_selectivo", "selectivo")

        total_muestras = len(datos)
        pct_usable_crudo = round(100 * (datos[usab_crudo] == "OK").sum() / total_muestras, 2) if total_muestras else float("nan")
        pct_usable_total = round(100 * (datos[usab_total] == "OK").sum() / total_muestras, 2) if total_muestras else float("nan")
        pct_usable_selectivo = round(100 * (datos[usab_selectivo] == "OK").sum() / total_muestras, 2) if total_muestras else float("nan")

        comp_origen = datos["origen_dato"].value_counts(normalize=True) * 100
        pct_medido_ok = round(comp_origen.get("medido_ok", 0.0), 2)
        pct_medido_flag = round(comp_origen.get("medido_flag", 0.0), 2)
        pct_sin_dato = round(comp_origen.get("sin_dato", 0.0), 2)

        print(f"\n   --- Composicion de la serie ({clave}, {FREQ}) ---")
        print(f"      medido_ok (dato real y valido):     {pct_medido_ok:.2f}%")
        print(f"      medido_flag (dato real, corregido):  {pct_medido_flag:.2f}%")
        print(f"      sin_dato (hueco, rellenado):          {pct_sin_dato:.2f}%")
        print(f"\n   --- %% usable (regla base QC) ---")
        print(f"      Crudo:                {pct_usable_crudo:.2f}%")
        print(f"      Corregido (total):    {pct_usable_total:.2f}%")
        print(f"      Corregido (selectivo): {pct_usable_selectivo:.2f}%")

        for variante, usab_col, pct_usable in [
            ("total", usab_total, pct_usable_total),
            ("selectivo", usab_selectivo, pct_usable_selectivo),
        ]:
            filas_comparativa.append({
                "estacion": clave,
                "variante": variante,
                "resolucion": FREQ,
                "modelo_ganador": ganador,
                "RMSE_test": round(float(rmse_ganador), 2),
                "pct_usable_crudo": pct_usable_crudo,
                "pct_usable_corregido": pct_usable,
                "mejora_pct_usable_pp": round(pct_usable - pct_usable_crudo, 2),
                "pct_medido_ok": pct_medido_ok,
                "pct_medido_flag_corregido": pct_medido_flag,
                "pct_sin_dato_rellenado": pct_sin_dato,
            })

        # --- Guardar serie corregida compacta para el dashboard ---
        salida = datos[[
            "datetime", col_crudo, "GHI_corregido_total", "GHI_corregido_selectivo",
            "origen_dato", usab_crudo, usab_total, usab_selectivo,
        ]].rename(columns={col_crudo: "GHI_original"})
        report.guardar_dataset(salida, f"{clave}_ghi_corregido.csv", etapa="processed")

        # --- Tablas de banderas detalladas por serie (opcional, para inspeccion) ---
        res_crudo = ev.resumen_flags_serie(datos, cols_crudo, "crudo")
        res_total = ev.resumen_flags_serie(datos, cols_total, "total")
        res_selectivo = ev.resumen_flags_serie(datos, cols_selectivo, "selectivo")
        tabla_flags = pd.concat([res_crudo, res_total, res_selectivo], ignore_index=True)
        tabla_flags.insert(0, "estacion", clave)
        report.guardar_tabla(tabla_flags, f"reconstruccion_{clave}_flags_comparativo.csv")

        # --- Tabla de metricas de todos los modelos (incluye la columna de
        #     validez fisica sobre el periodo de aplicacion) ---
        # Se acumula en todas_metricas y se guarda una sola vez fuera del
        # ciclo, para que el CSV contenga las filas de todas las estaciones
        # y no solo las de la ultima procesada.
        tabla_metricas_out = tabla_metricas.copy()
        tabla_metricas_out.insert(0, "estacion", clave)
        tabla_metricas_out["modelo_ganador"] = tabla_metricas_out["modelo"] == ganador
        todas_metricas.append(tabla_metricas_out)

    if todas_metricas:
        report.guardar_tabla(pd.concat(todas_metricas, ignore_index=True), "reconstruccion_metricas_modelos.csv")

    if filas_comparativa:
        tabla_comparativa = pd.DataFrame(filas_comparativa)
        print("\n" + "=" * 72)
        print(" TABLA COMPARATIVA FINAL (fuente(s) terrestre(s) x 2 variantes)")
        print("=" * 72)
        print(tabla_comparativa.to_string(index=False))
        report.guardar_tabla(tabla_comparativa, "reconstruccion_comparativo_4estaciones.csv")

    print("\n" + "=" * 72)
    print(" PIEZA 7b LISTA.")
    print(" Revisa outputs/tablas/reconstruccion_comparativo_4estaciones.csv y")
    print(" data/processed/{estacion}_ghi_corregido.csv para cada estacion.")
    print("=" * 72)


if __name__ == "__main__":
    main()
