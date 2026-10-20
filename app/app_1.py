"""
app.py — Dashboard Streamlit del proyecto de control de calidad (QC) y
adaptación al sitio (site-adaptation) del recurso solar.

Ejecución:
    cd qc-siteadapt-solar/
    streamlit run app/app.py

Las páginas Inicio/OE1/OE2/OE3 solo presentan resultados ya calculados:
leen tablas y figuras de outputs/ y datos de data/ sin volver a correr el
pipeline.

La página "Nuevo dataset (demo)" es la excepción: ejecuta en vivo el
pipeline completo OE1 -> OE2 -> OE3, con el código de src/qcsolar, sobre
conjuntos de datos que sube el usuario. Así se demuestra que la
metodología se puede aplicar a cualquier sitio y no solo a las fuentes
del caso de estudio. Requiere pvlib, scikit-learn y pyyaml (ver
requirements_app.txt).
"""

from pathlib import Path
import base64
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Descarga habilitada en TODAS las gráficas Plotly de la app: se envuelve
# st.plotly_chart una sola vez aquí (en lugar de modificar cada llamado)
# para que el botón de cámara del modebar (▸ descargar como PNG)
# quede siempre visible, sin el logo de Plotly, y en alta resolución.
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
_st_plotly_chart_original = st.plotly_chart

def _plotly_chart_con_descarga(*args, **kwargs):
    """Reemplazo de st.plotly_chart que inyecta PLOTLY_CONFIG cuando el
    llamador no define su propio 'config'.

    Ajusta además el título para que no choque con el modebar (íconos fijos
    arriba a la derecha):
    - Si la gráfica tiene título, se alinea a la izquierda (x=0); centrado,
      un título largo en un panel angosto queda debajo de los íconos.
    - Si no tiene título, se fuerza texto vacío: un título ausente hace que
      Plotly.js muestre el texto literal "undefined" sobre el rangeselector.
    update_layout combina atributos, así que se conservan el color y el
    tamaño de fuente que haya definido cada gráfica."""
    kwargs.setdefault("config", PLOTLY_CONFIG)
    if args and hasattr(args[0], "update_layout"):
        fig = args[0]
        try:
            titulo_actual = fig.layout.title.text
        except Exception:
            titulo_actual = None
        if titulo_actual:
            fig.update_layout(title=dict(text=titulo_actual, x=0, xanchor="left"))
        else:
            fig.update_layout(title=dict(text=""))
    return _st_plotly_chart_original(*args, **kwargs)

st.plotly_chart = _plotly_chart_con_descarga

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — adaptar si la app vive en otra carpeta relativa al proyecto
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # raiz del proyecto
DIR_FIGURAS = ROOT / "outputs" / "figuras"
DIR_TABLAS = ROOT / "outputs" / "tablas"
DIR_PROCESSED = ROOT / "data" / "processed"

# ─────────────────────────────────────────────────────────────────────────────
# QCSOLAR EN VIVO — para la pestaña "Nuevo dataset" (aplicar el mismo
# pipeline OE1+OE2+OE3 a datasets subidos por el usuario, no precalculados).
# El resto de la app no importa qcsolar: solo lee los CSV que generan los
# scripts de verificación del proyecto. Este bloque es la única excepción.
# Se protege con try/except porque pvlib/scikit-learn/pyyaml son opcionales:
# sin ellos la app sigue funcionando en modo de solo lectura.
# ─────────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(ROOT / "src"))

QCSOLAR_DISPONIBLE = True
QCSOLAR_ERROR = None
try:
    import pvlib
    from qcsolar.config import CONFIG as QCSOLAR_CONFIG
    from qcsolar.qc import physical as qcf_live
    from qcsolar.qc import expert as qce_live
    from qcsolar.qc import usability as qcu_live
    from qcsolar.siteadapt import prepare as sap_live
    from qcsolar.siteadapt import models as sam_live
    from qcsolar.siteadapt import evaluate as sae_live
    from qcsolar import metrics as met_live
except Exception as _e_qcsolar:
    QCSOLAR_DISPONIBLE = False
    QCSOLAR_ERROR = f"{type(_e_qcsolar).__name__}: {_e_qcsolar}"


def agregar_geometria_solar_generico(df: pd.DataFrame, lat: float, lon: float,
                                      alt: float, tz: str,
                                      datetime_col: str = "datetime") -> pd.DataFrame:
    """Variante parametrizada de qcsolar.solar.agregar_geometria_solar().
    Aquella toma lat/lon/altitud/zona horaria de config.yaml (sitio del caso
    de estudio); esta recibe esos valores como parámetros y hace el mismo
    cálculo con pvlib (posición solar, irradiancia extraterrestre y cielo
    claro Ineichen), con las mismas columnas de salida, para cualquier sitio."""
    df = df.copy()
    tiempos = pd.DatetimeIndex(df[datetime_col])
    if tiempos.tz is None:
        tiempos = tiempos.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        tiempos = tiempos.tz_convert(tz)

    posicion = pvlib.solarposition.get_solarposition(tiempos, lat, lon, altitude=alt)
    df["zenith"] = posicion["zenith"].to_numpy()
    df["elevation"] = posicion["elevation"].to_numpy()
    df["azimuth"] = posicion["azimuth"].to_numpy()
    df["es_de_dia"] = df["elevation"] > 0
    df["es_de_noche"] = df["zenith"] > 90
    df["dni_extra"] = pvlib.irradiance.get_extra_radiation(tiempos).to_numpy()

    sitio = pvlib.location.Location(lat, lon, tz=tz, altitude=alt, name="sitio_nuevo")
    cielo_claro = sitio.get_clearsky(tiempos, model="ineichen")
    df["ghi_clear"] = cielo_claro["ghi"].to_numpy()
    df["dni_clear"] = cielo_claro["dni"].to_numpy()
    df["dhi_clear"] = cielo_claro["dhi"].to_numpy()
    return df


def _leer_csv_generico(uploaded_file) -> "pd.DataFrame | None":
    """Lee un CSV subido por el usuario probando separadores comunes
    (coma, punto y coma, tab). Devuelve None si ninguno produce mas de
    1 columna util."""
    if uploaded_file is None:
        return None
    mejor = None
    for sep in [",", ";", "\t"]:
        try:
            uploaded_file.seek(0)
            df_intento = pd.read_csv(uploaded_file, sep=sep, engine="python")
            if df_intento.shape[1] > 1 and (mejor is None or df_intento.shape[1] > mejor.shape[1]):
                mejor = df_intento
        except Exception:
            continue
    return mejor


def _parsear_fecha_generica(serie: pd.Series) -> "tuple[pd.Series, str | None]":
    """Convierte una columna cruda a datetime, con manejo explícito de las
    fechas puramente numéricas tipo YYYYMMDD[HH[MM]] (p. ej. 2018010100).
    Pasadas tal cual a pandas.to_datetime se interpretan como nanosegundos
    desde 1970-01-01 sin error ni NaT, lo que produce fechas erróneas; por
    eso el patrón se detecta por longitud y se parsea con formato explícito.

    Devuelve (fechas_parseadas, formato_detectado_o_None).
    """
    texto = serie.astype(str).str.strip()
    solo_digitos = texto.str.fullmatch(r"\d{8}|\d{10}|\d{12}")
    fmt = None
    if solo_digitos.fillna(False).mean() > 0.9:
        largo = int(texto.str.len().median())
        fmt = {8: "%Y%m%d", 10: "%Y%m%d%H", 12: "%Y%m%d%H%M"}.get(largo)

    if fmt:
        fechas = pd.to_datetime(texto, format=fmt, errors="coerce")
    else:
        fechas = pd.to_datetime(texto, errors="coerce")
    return fechas, fmt


def _fraccion_fechas_sanas(fechas: pd.Series) -> float:
    """Fraccion de fechas no nulas que caen en un rango fisicamente
    razonable (1990 a hoy+2 anios). Sirve para detectar el caso silencioso
    de pandas interpretando una columna numerica como nanosegundos desde
    epoch (el resultado siempre cae en 1970, aunque "parseo" sin error)."""
    validas = fechas.dropna()
    if len(validas) == 0:
        return 0.0
    tz = validas.dt.tz
    hoy = pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now()
    limite_inf = pd.Timestamp("1990-01-01", tz=tz)
    limite_sup = hoy + pd.Timedelta(days=730)
    sanas = (validas >= limite_inf) & (validas <= limite_sup)
    return float(sanas.mean())


def _detectar_muestras_por_hora(datetime_series: pd.Series) -> float:
    """Estima cuántas muestras por hora trae una serie (5 min -> 12/h,
    horaria -> 1/h, etc.) a partir de la mediana del intervalo entre
    timestamps consecutivos. Evita asumir una resolución fija: si se estima
    mal, la cobertura mínima horaria nunca se cumple y el emparejamiento
    queda vacío."""
    t = pd.Series(datetime_series).dropna().sort_values().drop_duplicates()
    if len(t) < 3:
        return 1.0
    delta_seg = t.diff().dt.total_seconds().dropna()
    delta_mediana = delta_seg.median()
    if not delta_mediana or delta_mediana <= 0:
        return 1.0
    return max(1.0, round(3600.0 / delta_mediana))


def ejecutar_pipeline_generico(df_sat_std: pd.DataFrame, df_terr_std: pd.DataFrame,
                                lat: float, lon: float, alt: float, tz: str,
                                nombre_objetivo: str = "nuevo_sitio",
                                periodo_inicio: str | None = None,
                                periodo_fin: str | None = None) -> dict:
    """Orquesta el pipeline COMPLETO OE1 -> OE2 -> OE3 sobre dos datasets
    ya normalizados al esquema estandar (columna 'datetime' tz-aware +
    'GHI' [+'DNI'/'DHI' si hay] en W/m2), reusando el mismo codigo de
    qcsolar (src/qcsolar) que procesa el caso de estudio: ningún umbral ni
    criterio se reimplementa aquí, solo se invoca con los parámetros del
    sitio nuevo.

    `periodo_inicio`/`periodo_fin` (str "AAAA-MM-DD", opcionales): recortan
    el periodo usado para ENTRENAR/EVALUAR (conjunto "supervisado"), tal
    como elige el usuario en la UI. Si no se dan, se usa automaticamente
    todo el traslape de fechas entre ambas fuentes. `aplicacion` (la
    reconstruccion completa) no se recorta: cubre todo el rango donde la
    referencia satelital tiene dato.

    Devuelve un dict con todos los resultados intermedios, listos para
    graficar en la pestaña "Nuevo dataset".
    """
    resultado = {"advertencias": []}

    # ── 1) Geometria solar parametrizada con las coordenadas del sitio ──
    df_sat = agregar_geometria_solar_generico(df_sat_std, lat, lon, alt, tz)
    df_terr = agregar_geometria_solar_generico(df_terr_std, lat, lon, alt, tz)

    # ── 2) QC fisico + experto (OE1) — codigo generico de qcsolar.qc ──
    tiene_3_componentes = {"DNI", "DHI"}.issubset(df_sat.columns)
    fuente_sat = "solcast" if tiene_3_componentes else "satelital_nuevo"

    _alt_original = qce_live.ALT
    qce_live.ALT = alt  # test_ktests usa la altitud del sitio (Kn depende de ALT)
    try:
        df_sat = qcf_live.aplicar_qc_fisico(df_sat, fuente=fuente_sat)
        df_sat = qce_live.aplicar_qc_experto(df_sat, fuente=fuente_sat)
        df_sat["usabilidad"] = qcu_live.calcular_usabilidad(df_sat).to_numpy()

        df_terr = qcf_live.aplicar_qc_fisico(df_terr, fuente="terrestre_nuevo")
        df_terr = qce_live.aplicar_qc_experto(df_terr, fuente="terrestre_nuevo")
        df_terr["usabilidad"] = qcu_live.calcular_usabilidad(df_terr).to_numpy()
    finally:
        qce_live.ALT = _alt_original  # restaurar siempre la altitud configurada del caso de estudio

    resultado["df_sat_qc"] = df_sat
    resultado["df_terr_qc"] = df_terr
    resultado["fuente_sat"] = fuente_sat

    # ── 3) Emparejamiento horario generico (OE2, paso 1) ──
    muestras_h_sat = _detectar_muestras_por_hora(df_sat["datetime"])
    inicio_datos = max(df_sat["datetime"].min(), df_terr["datetime"].min())
    fin_datos = min(df_sat["datetime"].max(), df_terr["datetime"].max())
    if pd.isna(inicio_datos) or pd.isna(fin_datos) or inicio_datos >= fin_datos:
        resultado["error"] = ("Los dos datasets no tienen un periodo de fechas en comun. "
                               "Revisa el mapeo de columnas de fecha/hora.")
        return resultado

    # Periodo de entrenamiento/evaluacion: el que eligio el usuario en la UI
    # (ya acotado por ella al traslape real entre las 2 fuentes), o si no se
    # dio, el traslape completo detectado automaticamente.
    periodo_inicio_final = periodo_inicio or str(inicio_datos.date())
    periodo_fin_final = periodo_fin or str(fin_datos.date())

    supervisado, aplicacion = sap_live.emparejar_horario_generico(
        df_referencia=df_sat, df_objetivo=df_terr, nombre_objetivo=nombre_objetivo,
        col_objetivo="GHI",
        periodo_inicio=periodo_inicio_final, periodo_fin=periodo_fin_final,
        columna_usabilidad="usabilidad",
        muestras_esperadas_referencia=muestras_h_sat,
    )
    resultado["muestras_h_sat"] = muestras_h_sat
    resultado["periodo_inicio"] = periodo_inicio_final
    resultado["periodo_fin"] = periodo_fin_final
    resultado["supervisado"] = supervisado
    resultado["aplicacion"] = aplicacion

    if supervisado.empty:
        resultado["error"] = (
            f"No quedaron horas supervisadas dentro del periodo "
            f"{periodo_inicio_final} a {periodo_fin_final}. Elige un periodo más amplio "
            f"o confirma que ambos datasets tengan datos ahí."
        )
        return resultado

    col_objetivo = f"GHI_{nombre_objetivo}"
    if len(supervisado) < 30:
        resultado["error"] = (
            f"Solo se emparejaron {len(supervisado)} horas confiables entre los dos "
            f"datasets (se necesitan al menos ~30 para entrenar/evaluar algo razonable). "
            f"Revisa que las fechas se solapen y que el mapeo de columnas sea correcto."
        )
        return resultado

    # ── 4) Split temporal 70/30 (OE2, paso 2) ──
    train, test = sap_live.split_temporal(supervisado)
    resultado["train"] = train
    resultado["test"] = test
    resultado["resumen_emparejamiento"] = sap_live.resumen_emparejamiento_generico(
        supervisado, aplicacion, nombre_objetivo, train, test)

    if len(test) < 10:
        resultado["error"] = (f"Solo quedaron {len(test)} horas en el conjunto de prueba (30%) "
                               f"— muy pocas para evaluar modelos con confianza.")
        return resultado

    # ── 5) Entrenar los 9 modelos y evaluar SOLO en test (OE2, paso 3) ──
    variables_ref = sap_live.VARIABLES_SOLCAST_PREDICTORAS
    columnas_predictoras = [f"{v}_solcast" for v in variables_ref
                            if f"{v}_solcast" in supervisado.columns] + ["hour", "dayofyear", "month"]
    tabla_metricas, predicciones_test, modelos_entrenados = sae_live.entrenar_y_evaluar(
        train, test, target=col_objetivo, columnas_predictoras=columnas_predictoras)
    resultado["columnas_predictoras"] = columnas_predictoras
    resultado["tabla_metricas"] = tabla_metricas
    resultado["predicciones_test"] = predicciones_test
    resultado["modelos_entrenados"] = modelos_entrenados

    # ── 6) Elegir modelo ganador (mismo criterio de OE2) ──
    minimo_fisico = QCSOLAR_CONFIG["qc_fisico"]["minimo_irradiancia"]
    ganador, mejor_rmse_puro, tabla_enriquecida = sae_live.elegir_mejor_modelo(
        tabla_metricas, predicciones_test, target=col_objetivo, minimo_fisico=minimo_fisico)
    resultado["ganador"] = ganador
    resultado["mejor_rmse_puro"] = mejor_rmse_puro
    resultado["tabla_metricas_enriquecida"] = tabla_enriquecida

    # ── 7) Construir serie corregida (total + selectiva) sobre TODA la aplicacion ──
    modelo_ganador = modelos_entrenados[ganador]
    df_corregido = sae_live.construir_serie_corregida(aplicacion, modelo_ganador, nombre_objetivo)
    resultado["df_corregido"] = df_corregido

    # ── 8) Re-QC (OE3): geometria propia (sin sufijo) + mismo QC de OE1 ──
    df_corregido_geo = agregar_geometria_solar_generico(df_corregido, lat, lon, alt, tz)

    qce_live.ALT = alt
    try:
        df_reqc_total, cols_flag_total, col_usab_total = sae_live.aplicar_qc_a_serie(
            df_corregido_geo, col_ghi="GHI_corregido_total", prefijo="total")
        df_reqc, cols_flag_sel, col_usab_sel = sae_live.aplicar_qc_a_serie(
            df_reqc_total, col_ghi="GHI_corregido_selectivo", prefijo="selectivo")
    finally:
        qce_live.ALT = _alt_original

    resultado["df_reqc"] = df_reqc
    resultado["col_usab_total"] = col_usab_total
    resultado["col_usab_sel"] = col_usab_sel

    # % usable crudo (dato terrestre original, en las horas que tambien
    # tienen referencia satelital valida) vs. % usable de cada variante corregida
    pct_usable_crudo = float((df_reqc["origen_dato"] == "medido_ok").mean() * 100)
    resultado["pct_usable_crudo"] = pct_usable_crudo
    resultado["pct_usable_total"] = float((df_reqc[col_usab_total] == "OK").mean() * 100)
    resultado["pct_usable_selectivo"] = float((df_reqc[col_usab_sel] == "OK").mean() * 100)

    return resultado

# colores del proyecto
C_TEAL   = "#1F4E5F"
C_BLUE   = "#2b6f8f"
C_RED    = "#d1495b"
C_GOLD   = "#e8a838"
C_LIGHT  = "#f0f6f9"
C_WHITE  = "#ffffff"
# Azul de alto contraste (vs. C_RED) para los diagramas esquemáticos del
# marco metodológico OE2 -- C_TEAL/C_BLUE son muy parecidos entre sí para
# distinguir dos series/categorías en el mismo gráfico.
C_AZUL_VIVO = "#1565C0"

# Colores fijos (independientes del color de cada estación) para las
# gráficas "crudo vs. corregido (total/selectivo)" de OE2, para que ambas
# variantes no se confundan con el color propio de cada estación.
COLOR_CORREGIDO_TOTAL      = C_TEAL       # azul-teal oscuro, siempre igual
COLOR_CORREGIDO_SELECTIVO  = "#F4A261"    # naranja, contraste alto vs. teal y gris

FUENTES_LABEL = {
    "solcast":       "Solcast 2023 (satelital)",
    "udenar":        "Udenar (terrestre, 2023)",
    "solcast_2025":  "Solcast 2025 (satelital)",
    "udenar_est":    "Udenar (estación real, 2025)",
}

# Segundo caso de estudio: estación meteorológica terrestre con su propia
# referencia satelital. Es un conjunto más ruidoso que el primero: presenta
# un quiebre de régimen documentado (ver config.yaml) y activa el test de
# sensor caído (trae TEMP/RH). Constantes centralizadas para reutilizar el
# mismo nombre/color en toda la app.
CLAVE_UDENAR_EST = "udenar_est"
NOMBRE_UDENAR_EST = "Udenar (estación 2025)"
C_UDENAR_EST = C_GOLD

# Escala de color para heatmaps/envolventes (día×hora, envolvente solar).
# Paleta propia del proyecto: claro (poca irradiancia) -> teal -> dorado
# (irradiancia alta). Se evita negro y rosado como color de fondo/valor bajo.
ESCALA_CALOR_SOLAR = [
    [0.00, "#f7fbff"],
    [0.12, "#c6dbef"],
    [0.28, "#6baed6"],
    [0.44, "#3288bd"],
    [0.58, "#66c2a5"],
    [0.70, "#ffffbf"],
    [0.82, "#fdae61"],
    [0.92, "#f46d43"],
    [1.00, "#d73027"],
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fig_path(nombre: str) -> Path:
    """Ruta absoluta a una figura PNG guardada en outputs/figuras/."""
    return DIR_FIGURAS / nombre

def tabla_path(nombre: str) -> Path:
    """Ruta absoluta a una tabla CSV guardada en outputs/tablas/."""
    return DIR_TABLAS / nombre

def cargar_csv(nombre: str, directorio: Path = DIR_TABLAS) -> pd.DataFrame | None:
    """Lee un CSV de `directorio` (por defecto outputs/tablas/); None si no existe."""
    p = directorio / nombre
    if not p.exists():
        return None
    return pd.read_csv(p)

def mostrar_figura(nombre: str, caption: str = "", ancho: int | None = None) -> bool:
    """Muestra una figura PNG de outputs/figuras/ en la app; avisa si falta el archivo."""
    p = fig_path(nombre)
    if not p.exists():
        st.warning(f"Figura no encontrada: `{nombre}`")
        return False
    img = Image.open(p)
    if ancho is None:
        st.image(img, caption=caption or None, width="stretch")
    else:
        st.image(img, caption=caption or None, width=ancho)
    return True

def boton_descarga_fig(nombre: str, label: str = "⬇ Descargar figura"):
    """Botón de Streamlit para descargar una figura PNG estática (no Plotly)."""
    p = fig_path(nombre)
    if not p.exists():
        return
    with open(p, "rb") as f:
        st.download_button(label, f.read(), file_name=nombre, mime="image/png", key=f"dl_{nombre}")

def boton_descarga_csv(df: pd.DataFrame, nombre: str, label: str = "⬇ Descargar tabla (.csv)"):
    """Botón de Streamlit para descargar cualquier DataFrame como CSV."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv_bytes, file_name=nombre, mime="text/csv", key=f"dl_{nombre}")

@st.cache_data(show_spinner="Cargando datos procesados...")
def cargar_procesado(nombre: str) -> pd.DataFrame | None:
    """Carga un CSV de data/processed/ (predicciones, datos con geometria solar, etc.)."""
    p = DIR_PROCESSED / nombre
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if getattr(df["datetime"].dt, "tz", None) is not None:
            # el offset ya es la hora local del sitio (-05:00); solo se quita
            # la etiqueta de tz para que sea consistente con el resto de la app
            df["datetime"] = df["datetime"].dt.tz_localize(None)
    return df

def _usabilidad_desde_flags(df: pd.DataFrame) -> pd.Series:
    """Reconstruye la usabilidad (OK / NO_USABLE) a partir de las columnas
    flag_* de un CSV *_qc_completo.csv, sin importar qcsolar. Aplica la misma
    regla que qcsolar.qc.usability.calcular_usabilidad: NO_USABLE si alguna
    prueba evaluable marca FLAG; las pruebas de solo diagnóstico (simetría)
    no cuentan."""
    cols_flag = [c for c in df.columns if c.startswith("flag_") and "simetria" not in c]
    if not cols_flag:
        return pd.Series("OK", index=df.index)
    es_flag = (df[cols_flag] == "FLAG").any(axis=1)
    return pd.Series(np.where(es_flag, "NO_USABLE", "OK"), index=df.index)

_ETAPAS_DIAGNOSTICO = [
    ("crudo", "① Crudo"),
    ("post_qc", "② Post-QC inicial"),
    ("post_reqc", "③ Post re-QC (site-adaptation)"),
]

@st.cache_data(show_spinner=False)
def _construir_df_etapa_diagnostico(clave: str, etapa: str) -> pd.DataFrame | None:
    """Arma un DataFrame con datetime/GHI/geometría solar (azimuth, elevation,
    ghi_clear) para una etapa de depuración de una estación, de modo que las
    mismas funciones de gráfico (heatmap, CSI, envolvente, simetría AM/PM)
    sirvan para comparar antes/después en el mosaico:

      "crudo"      -- GHI tal cual llega del sensor, sin filtrar (incluye
                      las horas que el QC descarta más adelante).
      "post_qc"    -- GHI del control de calidad inicial (físico + experto):
                      NaN en las horas que no pasan.
      "post_reqc"  -- GHI ya corregido por site-adaptation (variante
                      "total"), con NaN en las horas que no pasan el
                      re-control de calidad final.
      "site_adapt" -- resultado directo de site-adaptation (variante
                      "total"), SIN el filtro del re-QC final -- es el
                      resultado propio de esta etapa (OE2); el re-QC es
                      responsabilidad de la etapa de verificación (OE3).

    Se trabaja a la resolución nativa de 5 min (ver marco metodológico de
    OE2); "post_reqc"/"site_adapt" leen `{clave}_ghi_corregido.csv`, que ya
    viene a esa resolución.

    Devuelve None si falta el archivo fuente. No depende de qcsolar/sklearn:
    solo reusa columnas y flags guardados en los CSV de data/processed/."""
    df_qc = cargar_procesado(f"{clave}_qc_completo.csv")
    if df_qc is None:
        return None

    if etapa == "crudo":
        return df_qc

    if etapa == "post_qc":
        d = df_qc.copy()
        usab = _usabilidad_desde_flags(d)
        d["GHI"] = d["GHI"].where(usab.values == "OK")
        return d

    if etapa in ("post_reqc", "site_adapt"):
        df_corr = cargar_procesado(f"{clave}_ghi_corregido.csv")
        if df_corr is None or "GHI_corregido_total" not in df_corr.columns:
            return None
        if etapa == "post_reqc" and "usabilidad_total" not in df_corr.columns:
            return None
        cols_geo = [c for c in ["datetime", "azimuth", "elevation", "ghi_clear"] if c in df_qc.columns]
        # `{clave}_ghi_corregido.csv` y `{clave}_qc_completo.csv` deberían
        # estar en la MISMA rejilla nativa de 5 min; se usa merge_asof
        # (vecino más cercano, tolerancia 15 min) para tolerar instantes
        # desalineados, que con un merge exacto dejarían azimut/elevación
        # en NaN.
        d_corr = df_corr.sort_values("datetime")
        d_geo = df_qc[cols_geo].sort_values("datetime")
        d = pd.merge_asof(d_corr, d_geo, on="datetime", direction="nearest",
                           tolerance=pd.Timedelta("15min"))
        if etapa == "post_reqc":
            d["GHI"] = d["GHI_corregido_total"].where(d["usabilidad_total"] == "OK")
        else:
            d["GHI"] = d["GHI_corregido_total"]
        return d

    return None

def _metricas_regresion_simple(y_real: pd.Series, y_pred: pd.Series) -> dict:
    """Calcula MAE, RMSE, MBE, rMBE, rRMSE, Pearson r, R2 y n con las mismas
    fórmulas que qcsolar.metrics.metricas_regresion, pero con numpy/pandas
    puro para no depender de sklearn en las páginas de solo lectura."""
    y_real = pd.Series(y_real).astype(float).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(float).reset_index(drop=True)
    validos = pd.concat([y_real, y_pred], axis=1).dropna()
    validos.columns = ["y_real", "y_pred"]
    n = len(validos)
    if n < 2:
        return {"MAE": np.nan, "RMSE": np.nan, "MBE": np.nan, "rMBE": np.nan,
                "rRMSE": np.nan, "Pearson_R": np.nan, "R2": np.nan, "n": n}
    err = validos["y_pred"] - validos["y_real"]
    mae = err.abs().mean()
    rmse = float(np.sqrt((err ** 2).mean()))
    mbe = err.mean()
    media_ref = validos["y_real"].mean()
    pearson = (validos["y_real"].corr(validos["y_pred"])
               if validos["y_real"].std() > 0 and validos["y_pred"].std() > 0 else np.nan)
    ss_tot = ((validos["y_real"] - media_ref) ** 2).sum()
    r2 = 1 - (err ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return {
        "MAE": mae, "RMSE": rmse, "MBE": mbe,
        "rMBE": 100 * mbe / media_ref if media_ref else np.nan,
        "rRMSE": 100 * rmse / media_ref if media_ref else np.nan,
        "Pearson_R": pearson, "R2": r2, "n": n,
    }

# Cada fuente terrestre tiene su propia fuente satelital de referencia
# emparejada (ver config.yaml: satelital_por_fuente); el valor es el prefijo
# del archivo *_qc_completo.csv de esa referencia. Para claves no listadas
# se usa la referencia satelital del primer caso de estudio.
SOLCAST_TAG_POR_FUENTE = {"udenar": "solcast", "udenar_est": "solcast_2025"}


@st.cache_data(show_spinner=False)
def calcular_metricas_qc_vs_solcast(clave: str) -> dict | None:
    """Compara el GHI de una estación terrestre contra su fuente satelital
    de referencia emparejada, en dos versiones: 'crudo' (todos los datos) y
    'post_qc' (solo horas que pasan el QC de la estación). Usa resolución
    horaria diurna (referencia > 10 W/m2). Devuelve un dict de métricas por
    versión, o None si faltan archivos."""
    _tag_sol = SOLCAST_TAG_POR_FUENTE.get(clave, "solcast")
    df_sol = cargar_procesado(f"{_tag_sol}_qc_completo.csv")
    df_est = cargar_procesado(f"{clave}_qc_completo.csv")
    if df_sol is None or df_est is None:
        return None
    if (not {"datetime", "GHI"}.issubset(df_sol.columns)
            or not {"datetime", "GHI"}.issubset(df_est.columns)):
        return None

    sol_h = (df_sol.dropna(subset=["datetime", "GHI"])
                    .set_index("datetime").sort_index()["GHI"]
                    .resample("1h").mean())

    est_idx = df_est.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    usab = _usabilidad_desde_flags(est_idx)
    est_crudo_h = est_idx["GHI"].resample("1h").mean()
    est_qc_h = est_idx["GHI"].mask(usab.values != "OK").resample("1h").mean()

    resultados = {}
    for etiqueta, serie_est in [("crudo", est_crudo_h), ("post_qc", est_qc_h)]:
        _df = pd.concat([sol_h.rename("solcast"), serie_est.rename("estacion")], axis=1).dropna()
        _df = _df[_df["solcast"] > 10]
        if _df.empty:
            resultados[etiqueta] = None
            continue
        m = _metricas_regresion_simple(_df["estacion"], _df["solcast"])
        m["n_horas"] = len(_df)
        m["periodo"] = f"{_df.index.min().date()} → {_df.index.max().date()}"
        resultados[etiqueta] = m
    return resultados


def _parsear_rango_test(texto) -> tuple[str, str] | None:
    """Parsea el string de la columna `rango_test` de
    reconstruccion_resumen_emparejamiento.csv (formato
    "('2025-06-01 08:00:00-05:00', '2025-07-20 23:00:00-05:00')") a un par
    (inicio, fin) de strings de fecha listos para pd.Timestamp(). None si
    no se puede parsear (columna vacía o formato inesperado)."""
    try:
        partes = str(texto).strip("()").split(", ")
        return partes[0].strip("'\""), partes[1].strip("'\"")
    except (IndexError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def calcular_metricas_referencia_ventana(clave: str, test_ini: str, test_fin: str) -> dict | None:
    """Igual que `calcular_metricas_qc_vs_solcast`, pero restringido a la
    ventana [test_ini, test_fin]: el mismo conjunto de prueba con el que se
    evaluaron los modelos de site-adaptation de la estación (columna
    `rango_test` de reconstruccion_resumen_emparejamiento.csv). Así los tres
    escalones (crudo / post-QC / site-adaptado) se comparan sobre las mismas
    fechas."""
    _tag_sol = SOLCAST_TAG_POR_FUENTE.get(clave, "solcast")
    df_sol = cargar_procesado(f"{_tag_sol}_qc_completo.csv")
    df_est = cargar_procesado(f"{clave}_qc_completo.csv")
    if df_sol is None or df_est is None:
        return None
    if (not {"datetime", "GHI"}.issubset(df_sol.columns)
            or not {"datetime", "GHI"}.issubset(df_est.columns)):
        return None

    # test_ini/test_fin vienen del CSV con offset (-05:00, hora local),
    # pero cargar_procesado() deja "datetime" siempre tz-naive -- hay que
    # quitar la tz aqui tambien para poder comparar contra el indice.
    ini = pd.Timestamp(test_ini).tz_localize(None)
    fin = pd.Timestamp(test_fin).tz_localize(None)

    sol_h = (df_sol.dropna(subset=["datetime", "GHI"])
                    .set_index("datetime").sort_index()["GHI"]
                    .resample("1h").mean())
    sol_h = sol_h[(sol_h.index >= ini) & (sol_h.index <= fin)]

    est_idx = df_est.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    est_idx = est_idx[(est_idx.index >= ini) & (est_idx.index <= fin)]
    usab = _usabilidad_desde_flags(est_idx)
    est_crudo_h = est_idx["GHI"].resample("1h").mean()
    est_qc_h = est_idx["GHI"].mask(usab.values != "OK").resample("1h").mean()

    resultados = {}
    for etiqueta, serie_est in [("crudo", est_crudo_h), ("post_qc", est_qc_h)]:
        _df = pd.concat([sol_h.rename("solcast"), serie_est.rename("estacion")], axis=1).dropna()
        _df = _df[_df["solcast"] > 10]
        if _df.empty:
            resultados[etiqueta] = None
            continue
        m = _metricas_regresion_simple(_df["estacion"], _df["solcast"])
        m["n_horas"] = len(_df)
        resultados[etiqueta] = m
    return resultados


@st.cache_data(show_spinner=False)
def calcular_rmse_adaptado_tras_reqc(clave: str, test_ini: str, test_fin: str) -> dict | None:
    """RMSE del dato adaptado (variante 'total') frente al GHI REAL medido
    (no la referencia satelital), en el conjunto de prueba, con y sin exigir que la
    predicción también pase el re-QC final de la etapa de verificación de
    desempeño. Responde: ¿el re-QC final descarta predicciones que ya eran
    malas (RMSE similar) o realmente mejora la precisión (RMSE baja)?

    Se usa solo la variante 'total' (predicción del modelo en TODAS las
    horas) porque 'selectivo' conserva el dato medido tal cual en las horas
    donde ya había medición válida -- que es exactamente el subconjunto
    que se puede evaluar aquí (origen_dato == medido_ok) -- así que
    comparar 'selectivo' contra sí mismo daría RMSE ≈ 0, no es informativo.
    """
    df = cargar_procesado(f"{clave}_ghi_corregido.csv")
    if df is None or not {"GHI_original", "GHI_corregido_total", "usabilidad_total",
                           "origen_dato", "datetime"}.issubset(df.columns):
        return None
    ini = pd.Timestamp(test_ini).tz_localize(None)
    fin = pd.Timestamp(test_fin).tz_localize(None)
    d = df[(df["datetime"] >= ini) & (df["datetime"] <= fin) & (df["origen_dato"] == "medido_ok")]
    if d.empty:
        return None
    m_sin_reqc = _metricas_regresion_simple(d["GHI_original"], d["GHI_corregido_total"])
    m_sin_reqc["n_horas"] = len(d)
    d_ok = d[d["usabilidad_total"] == "OK"]
    m_con_reqc = None
    if not d_ok.empty:
        m_con_reqc = _metricas_regresion_simple(d_ok["GHI_original"], d_ok["GHI_corregido_total"])
        m_con_reqc["n_horas"] = len(d_ok)
    return {"sin_reqc": m_sin_reqc, "con_reqc": m_con_reqc}

@st.cache_data(show_spinner=False)
def fig_cobertura_mensual_plotly(serie: pd.Series, nombre: str, color: str) -> "go.Figure":
    """Barras de % de cobertura mensual. El número esperado de registros se
    deriva del intervalo de muestreo típico, más estable que la mediana de
    registros por día cuando la fuente cambia de densidad."""
    s = serie.dropna().sort_index()
    deltas_min = s.index.to_series().diff().dt.total_seconds().div(60).dropna()
    intervalo = deltas_min[(deltas_min > 0) & (deltas_min < 120)].median()
    if not intervalo or intervalo <= 0:
        intervalo = 5.0
    esperado_dia = 1440.0 / intervalo

    df_m = pd.DataFrame({"n": 1}, index=s.index)
    df_m["mes"] = df_m.index.to_period("M").astype(str)
    df_m["dia"] = df_m.index.normalize()
    registros_por_dia = df_m.groupby("dia")["n"].sum()
    registros_por_dia = registros_por_dia.to_frame()
    registros_por_dia["mes"] = registros_por_dia.index.to_period("M").astype(str)
    cobertura_mensual = (
        registros_por_dia.groupby("mes")["n"].mean() / esperado_dia * 100
    ).clip(upper=100)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cobertura_mensual.index, y=cobertura_mensual.values,
        marker_color=color, opacity=0.85,
        text=[f"{v:.0f}%" for v in cobertura_mensual.values],
        textposition="outside",
        hovertemplate="%{x}<br>Cobertura: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=95, line_dash="dash", line_color="#888",
                  annotation_text="Referencia 95%", annotation_position="top left")
    fig.update_layout(
        title=dict(text=f"{nombre} — cobertura mensual", font=dict(size=13)),
        height=340, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=40, b=20, l=40, r=10), showlegend=False,
        xaxis=dict(tickfont=dict(size=9), tickangle=-45),
        yaxis=dict(title="Cobertura [%]", gridcolor="#f0f0f0", range=[0, 110]),
    )
    return fig

@st.cache_data(show_spinner=False)
def resumen_intervalo_serie(serie: pd.Series) -> dict:
    """Resumen compacto de resolucion temporal de una serie: intervalo
    tipico (mediana, en minutos), % de registros que respetan ese
    intervalo, N de registros y periodo cubierto. Se prefiere a un
    histograma de intervalos, poco informativo cuando la resolucion es casi
    constante (una sola barra ocupa toda la figura)."""
    s = serie.dropna().sort_index()
    if len(s) < 2:
        return {"intervalo_min": None, "pct_regular": None,
                "n_registros": len(s), "periodo": "—"}
    deltas_min = s.index.to_series().diff().dt.total_seconds().div(60).dropna()
    deltas_validos = deltas_min[(deltas_min > 0) & (deltas_min < 120)]
    intervalo = deltas_validos.median() if not deltas_validos.empty else None
    pct_regular = (
        100 * (deltas_validos.round(1) == round(intervalo, 1)).mean()
        if intervalo is not None else None
    )
    return {
        "intervalo_min": intervalo,
        "pct_regular": pct_regular,
        "n_registros": len(s),
        "periodo": f"{s.index.min().date()} → {s.index.max().date()}",
    }

@st.cache_data(show_spinner=False)
def fig_continuidad_temporal_plotly(serie: pd.Series, nombre: str, color: str) -> "go.Figure":
    """Heatmap dia-del-anio x hora local: 1 si existe registro, 0 si no."""
    s = serie.dropna()
    df_c = pd.DataFrame({"dia": s.index.dayofyear, "hora": s.index.hour, "n": 1})
    pres = df_c.groupby(["dia", "hora"])["n"].sum().unstack(fill_value=0)
    pres = pres.reindex(index=range(1, 367), columns=range(24), fill_value=0)
    presencia = (pres > 0).astype(int)

    fig = go.Figure(go.Heatmap(
        z=presencia.values.T, x=presencia.index, y=presencia.columns,
        colorscale=[[0, "#f0f0f0"], [1, color]], showscale=False,
        hovertemplate="Día %{x}<br>Hora %{y}h<br>%{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{nombre} — continuidad temporal (blanco = hueco de datos)", font=dict(size=13)),
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=40, b=20, l=40, r=10),
        xaxis=dict(title="Día del año", gridcolor="#f0f0f0"),
        yaxis=dict(title="Hora local", gridcolor="#f0f0f0", dtick=3),
    )
    return fig

@st.cache_data(show_spinner=False)
def fig_heatmap_diario_plotly(df: pd.DataFrame, col: str, nombre: str, colorscale=None) -> "go.Figure | None":
    """Heatmap dia-del-anio x hora local del valor medio de `col` (GHI/DNI)."""
    if df is None or col not in df.columns or "datetime" not in df.columns:
        return None
    if colorscale is None:
        colorscale = ESCALA_CALOR_SOLAR
    d = df.dropna(subset=["datetime", col]).copy()
    d["dia"] = d["datetime"].dt.dayofyear
    d["hora"] = d["datetime"].dt.hour
    piv = d.groupby(["hora", "dia"])[col].mean().unstack()
    piv = piv.reindex(index=range(24), columns=range(1, 367))
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index, colorscale=colorscale,
        zmin=0, colorbar=dict(title=f"{col} [W/m²]"),
        hovertemplate="Día %{x}<br>Hora %{y}h<br>" + col + ": %{z:.0f} W/m²<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{nombre} — patrón anual de {col} (día × hora)", font=dict(size=13)),
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=40, b=20, l=40, r=10),
        xaxis=dict(title="Día del año"),
        yaxis=dict(title="Hora local", dtick=3),
    )
    return fig

@st.cache_data(show_spinner=False)
def fig_heatmap_comparativo_plotly(dfs: tuple, etiquetas: tuple, col: str, nombre: str,
                                    colorscale=None, alto_por_fila: int = 190) -> "go.Figure | None":
    """Heatmap día del año × hora local de `col`, en UNA figura con N
    subplots apilados verticalmente (uno por etapa) compartiendo una sola
    escala/barra de color -- misma metodología que
    fig_envolvente_comparativo_plotly: sirve tanto para 2 etapas (OE1:
    crudo/post-QC) como para 3 (OE2: crudo/post-QC/site-adaptado)."""
    if colorscale is None:
        colorscale = ESCALA_CALOR_SOLAR

    def _pivot(df):
        """Matriz hora (24) x día del año (366) con la media de `col`; None si no hay datos."""
        if df is None or col not in df.columns or "datetime" not in df.columns:
            return None
        d = df.dropna(subset=["datetime", col]).copy()
        if d.empty:
            return None
        d["dia"] = d["datetime"].dt.dayofyear
        d["hora"] = d["datetime"].dt.hour
        piv = d.groupby(["hora", "dia"])[col].mean().unstack()
        return piv.reindex(index=range(24), columns=range(1, 367))

    pivots = [_pivot(df) for df in dfs]
    if all(p is None for p in pivots):
        return None

    zmax = max(
        (float(p.values[~pd.isna(p.values)].max()) for p in pivots
         if p is not None and p.size and (~pd.isna(p.values)).any()),
        default=None,
    )
    if zmax is None or not np.isfinite(zmax):
        return None

    n = len(pivots)
    fig = make_subplots(rows=n, cols=1, subplot_titles=etiquetas,
                         shared_xaxes=True, vertical_spacing=0.06)
    for i, piv in enumerate(pivots):
        if piv is None:
            continue
        fig.add_trace(go.Heatmap(
            z=piv.values, x=piv.columns, y=piv.index,
            coloraxis="coloraxis",
            hovertemplate="Día %{x}<br>Hora %{y}h<br>" + col + ": %{z:.0f} W/m²<extra></extra>",
        ), row=i + 1, col=1)
        fig.update_yaxes(title_text="Hora local", dtick=6, row=i + 1, col=1)
        if i == n - 1:
            fig.update_xaxes(title_text="Día del año", showticklabels=True, row=i + 1, col=1)
        else:
            fig.update_xaxes(showticklabels=False, row=i + 1, col=1)

    fig.update_layout(
        title=dict(text=f"{nombre} — patrón anual de {col} (día × hora)",
                    font=dict(size=13)),
        height=alto_por_fila * n + 50, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, b=20, l=50, r=10),
        coloraxis=dict(colorscale=colorscale, cmin=0, cmax=zmax,
                        colorbar=dict(title=f"{col} [W/m²]", thickness=14)),
    )
    return fig

@st.cache_data(show_spinner=False)
def fig_csi_anual_plotly(df: pd.DataFrame, nombre: str, color: str) -> "go.Figure | None":
    """CSI = GHI / GHI_clearsky, mediana diaria a lo largo del año."""
    if df is None or not {"GHI", "ghi_clear", "datetime"}.issubset(df.columns):
        return None
    d = df.dropna(subset=["datetime"]).copy()
    d = d[d["ghi_clear"] > 10]
    if d.empty:
        return None
    d["csi"] = d["GHI"] / d["ghi_clear"]
    d["fecha"] = d["datetime"].dt.normalize()
    csi_diario = d.groupby("fecha")["csi"].median()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=csi_diario.index, y=csi_diario.values, mode="lines",
        line=dict(color=color, width=1.2), name="CSI mediano diario",
        hovertemplate="%{x|%d %b %Y}<br>CSI: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#22AA44",
                  annotation_text="CSI = 1 (ideal cielo claro)", annotation_position="bottom right")
    fig.update_layout(
        title=dict(text=f"{nombre} — índice de cielo claro (CSI) diario", font=dict(size=13)),
        height=340, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=40, b=20, l=40, r=10), showlegend=False,
        xaxis=dict(title="Fecha", gridcolor="#f0f0f0"),
        yaxis=dict(title="CSI mediano diario", gridcolor="#f0f0f0"),
    )
    return fig

@st.cache_data(show_spinner=False)
def fig_csi_comparativo_plotly(df_crudo: "pd.DataFrame | None", df_postqc: "pd.DataFrame | None",
                                nombre: str) -> "go.Figure | None":
    """CSI mediano diario, crudo vs. post-QC, en UNA sola gráfica con 2
    trazas de colores de alto contraste (C_RED / C_AZUL_VIVO) para comparar
    directamente el efecto del control de calidad inicial."""
    def _serie(df):
        """Serie de CSI (GHI / GHI de cielo claro) mediano diario; None si no hay datos."""
        if df is None or not {"GHI", "ghi_clear", "datetime"}.issubset(df.columns):
            return None
        d = df.dropna(subset=["datetime"]).copy()
        d = d[d["ghi_clear"] > 10]
        if d.empty:
            return None
        d["csi"] = d["GHI"] / d["ghi_clear"]
        d["fecha"] = d["datetime"].dt.normalize()
        return d.groupby("fecha")["csi"].median()

    s_crudo, s_postqc = _serie(df_crudo), _serie(df_postqc)
    if s_crudo is None and s_postqc is None:
        return None

    fig = go.Figure()
    if s_crudo is not None:
        fig.add_trace(go.Scatter(
            x=s_crudo.index, y=s_crudo.values, mode="lines",
            line=dict(color=C_RED, width=1.1), opacity=0.85, name="Crudo",
            hovertemplate="%{x|%d %b %Y}<br>CSI crudo: %{y:.2f}<extra></extra>",
        ))
    if s_postqc is not None:
        fig.add_trace(go.Scatter(
            x=s_postqc.index, y=s_postqc.values, mode="lines",
            line=dict(color=C_AZUL_VIVO, width=1.3), name="Post-QC",
            hovertemplate="%{x|%d %b %Y}<br>CSI post-QC: %{y:.2f}<extra></extra>",
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#22AA44",
                  annotation_text="CSI = 1 (ideal cielo claro)", annotation_position="bottom right")
    fig.update_layout(
        title=dict(text=f"{nombre} — índice de cielo claro (CSI) diario", font=dict(size=13)),
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=45, b=20, l=40, r=10),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(title="Fecha", gridcolor="#f0f0f0"),
        yaxis=dict(title="CSI mediano diario", gridcolor="#f0f0f0"),
    )
    return fig

@st.cache_data(show_spinner=False)
def fig_csi_comparativo3_plotly(df_crudo: "pd.DataFrame | None", df_postqc: "pd.DataFrame | None",
                                 df_siteadapt: "pd.DataFrame | None", nombre: str) -> "go.Figure | None":
    """CSI mediano diario en UNA sola gráfica con 3 trazas de alto contraste:
    crudo (rojo), post-QC (azul) y site-adaptado (verde) -- el resultado de
    OE2 (site-adaptation), agregado a las 2 etapas ya usadas en OE1."""
    def _serie(df):
        """Serie de CSI (GHI / GHI de cielo claro) mediano diario; None si no hay datos."""
        if df is None or not {"GHI", "ghi_clear", "datetime"}.issubset(df.columns):
            return None
        d = df.dropna(subset=["datetime"]).copy()
        d = d[d["ghi_clear"] > 10]
        if d.empty:
            return None
        d["csi"] = d["GHI"] / d["ghi_clear"]
        d["fecha"] = d["datetime"].dt.normalize()
        return d.groupby("fecha")["csi"].median()

    s_crudo, s_postqc, s_sa = _serie(df_crudo), _serie(df_postqc), _serie(df_siteadapt)
    if s_crudo is None and s_postqc is None and s_sa is None:
        return None

    fig = go.Figure()
    if s_crudo is not None:
        fig.add_trace(go.Scatter(
            x=s_crudo.index, y=s_crudo.values, mode="lines",
            line=dict(color=C_RED, width=1.0), opacity=0.8, name="① Crudo",
            hovertemplate="%{x|%d %b %Y}<br>CSI crudo: %{y:.2f}<extra></extra>",
        ))
    if s_postqc is not None:
        fig.add_trace(go.Scatter(
            x=s_postqc.index, y=s_postqc.values, mode="lines",
            line=dict(color=C_AZUL_VIVO, width=1.1), opacity=0.85, name="② Post-QC",
            hovertemplate="%{x|%d %b %Y}<br>CSI post-QC: %{y:.2f}<extra></extra>",
        ))
    if s_sa is not None:
        fig.add_trace(go.Scatter(
            x=s_sa.index, y=s_sa.values, mode="lines",
            line=dict(color="#22AA44", width=1.3), name="③ Site-adaptado",
            hovertemplate="%{x|%d %b %Y}<br>CSI site-adaptado: %{y:.2f}<extra></extra>",
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#999",
                  annotation_text="CSI = 1 (ideal cielo claro)", annotation_position="bottom right")
    fig.update_layout(
        title=dict(text=f"{nombre} — CSI diario", font=dict(size=13)),
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=45, b=20, l=40, r=10),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(title="Fecha", gridcolor="#f0f0f0"),
        yaxis=dict(title="CSI mediano diario", gridcolor="#f0f0f0"),
    )
    return fig

_COLORES_ETAPAS_DIAG = [C_RED, C_AZUL_VIVO, "#22AA44", C_GOLD]

@st.cache_data(show_spinner=False)
def fig_csi_comparativo_n_plotly(dfs: tuple, etiquetas: tuple, nombre: str) -> "go.Figure | None":
    """CSI mediano diario en UNA sola gráfica con N trazas de alto
    contraste (una por etapa); generaliza fig_csi_comparativo3_plotly a N
    etapas (p. ej. la cuarta, ④ post re-QC, en OE3)."""
    def _serie(df):
        """Serie de CSI (GHI / GHI de cielo claro) mediano diario; None si no hay datos."""
        if df is None or not {"GHI", "ghi_clear", "datetime"}.issubset(df.columns):
            return None
        d = df.dropna(subset=["datetime"]).copy()
        d = d[d["ghi_clear"] > 10]
        if d.empty:
            return None
        d["csi"] = d["GHI"] / d["ghi_clear"]
        d["fecha"] = d["datetime"].dt.normalize()
        return d.groupby("fecha")["csi"].median()

    series = [_serie(df) for df in dfs]
    if all(s is None for s in series):
        return None

    fig = go.Figure()
    for i, s in enumerate(series):
        if s is None:
            continue
        color = _COLORES_ETAPAS_DIAG[i % len(_COLORES_ETAPAS_DIAG)]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color=color, width=1.0 + 0.15 * i), opacity=0.85,
            name=etiquetas[i],
            hovertemplate="%{x|%d %b %Y}<br>CSI " + etiquetas[i] + ": %{y:.2f}<extra></extra>",
        ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#999",
                  annotation_text="CSI = 1 (ideal cielo claro)", annotation_position="bottom right")
    fig.update_layout(
        title=dict(text=f"{nombre} — CSI diario",
                    font=dict(size=13), y=0.97, yanchor="top"),
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=80, b=20, l=40, r=10),
        legend=dict(orientation="h", y=1.16, x=0),
        xaxis=dict(title="Fecha", gridcolor="#f0f0f0"),
        yaxis=dict(title="CSI mediano diario", gridcolor="#f0f0f0"),
    )
    return fig

_FLAGS_EXPERTO = {
    "flag_cierre", "flag_ktests", "flag_csi",
    "flag_roc_GHI", "flag_persistencia_GHI", "flag_simetria_GHI",
}

# Pruebas de "inspección visual" del procedimiento de QC experto de la
# literatura: solo diagnóstico, NUNCA invalidan un timestamp ni un día
# completo. Debe reflejar exactamente
# PRUEBAS_SOLO_DIAGNOSTICO en qcsolar/qc/usability.py -- estas graficas
# recalculan su propia mascara "pasa QC" en vez de usar calcular_usabilidad()
# directamente, asi que el filtro se replica aqui a mano.
_PRUEBAS_SOLO_DIAGNOSTICO = ["simetria"]

@st.cache_data(show_spinner=False)
def fig_evolucion_qc_plotly(df: pd.DataFrame, nombre: str, color: str,
                             max_puntos: int = 8000) -> "go.Figure | None":
    """Evolución de GHI a través de las 3 etapas de depuración: crudo ->
    pasa QC físico -> pasa QC físico + experto (acumulativo). Las flags
    físicas/expertas se detectan automáticamente por nombre de columna,
    así que sirve para cualquier fuente."""
    if df is None or not {"GHI", "datetime"}.issubset(df.columns):
        return None
    d = df.dropna(subset=["datetime", "GHI"]).copy()
    if d.empty:
        return None
    n_total = len(d)
    step = max(1, n_total // max_puntos)
    if step > 1:
        d = d.iloc[::step].copy()

    flags_todas = [c for c in d.columns if c.startswith("flag_")
                   and not any(diag in c for diag in _PRUEBAS_SOLO_DIAGNOSTICO)]
    flags_exp = [c for c in flags_todas if c in _FLAGS_EXPERTO]
    flags_fis = [c for c in flags_todas if c not in _FLAGS_EXPERTO]

    def _ok(cols):
        """True donde NINGUNA de las columnas de flag en `cols` marcó FLAG."""
        m = pd.Series(True, index=d.index)
        for c in cols:
            m &= (d[c] != "FLAG")
        return m

    mask_fis = _ok(flags_fis) if flags_fis else pd.Series(True, index=d.index)
    mask_exp = mask_fis & (_ok(flags_exp) if flags_exp else pd.Series(True, index=d.index))
    n = len(d)
    pct_fis = mask_fis.sum() / n * 100 if n else 0
    pct_exp = mask_exp.sum() / n * 100 if n else 0

    # Colores homogéneos con el resto de la app: el crudo usa el color de
    # la fuente (satelital=azul, terrestre=verde, vía el parámetro `color`);
    # QC físico=naranja y QC físico+experto=rojo, fijos, independientes de
    # la fuente (para que esas 2 etapas se lean igual en cualquier panel).
    _COLOR_QC_FISICO = "#E67E22"     # naranja
    _COLOR_QC_EXPERTO = C_RED        # rojo

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["datetime"], y=d["GHI"], mode="lines",
        line=dict(color=color, width=1),
        opacity=0.6,
        name="Crudo (100%)",
        hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d.loc[mask_fis, "datetime"], y=d.loc[mask_fis, "GHI"], mode="lines",
        line=dict(color=_COLOR_QC_FISICO, width=1.2),
        opacity=0.75,
        name=f"Pasa QC físico (~{pct_fis:.0f}%)",
        hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=d.loc[mask_exp, "datetime"], y=d.loc[mask_exp, "GHI"], mode="lines",
        line=dict(color=_COLOR_QC_EXPERTO, width=1.4),
        opacity=0.9,
        name=f"Pasa físico + experto (~{pct_exp:.0f}%)",
        hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{nombre} — evolución de la depuración (crudo → QC físico → QC experto)", font=dict(size=13)),
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=65, b=20, l=60, r=170),
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1, itemsizing="constant",
        ),
        xaxis=dict(
            showgrid=False, tickformat="%b %Y",
            rangeselector=dict(buttons=[
                dict(count=7,  label="1S",  step="day",   stepmode="backward"),
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(step="all", label="Todo"),
            ], bgcolor="#f0f6f9", activecolor="#1F4E5F", font=dict(size=10)),
            rangeslider=dict(visible=True, thickness=0.07),
        ),
        yaxis=dict(title=dict(text="GHI [W/m²]", font=dict(size=11)), gridcolor="#eeeeee", rangemode="tozero"),
    )
    return fig

@st.cache_data(show_spinner=False)
def fig_evolucion_reconstruccion_plotly(nombre_clave: str, nombre_display: str, color: str,
                                          variante: str = "total",
                                          max_puntos: int = 200_000,
                                          capas: tuple = ("solcast", "crudo", "crudo_qc",
                                                          "adapt_100", "adapt_qc")) -> "go.Figure | None":
    """Evolución completa de una estación terrestre a través de TODO el
    pipeline OE1->OE2->OE3: referencia satelital (sombreada) -> crudo (100%)
    -> crudo que pasa el QC de OE1 -> adaptado (site-adaptation, OE2) ->
    adaptado que pasa el re-QC de OE3. Muestra por qué el dataset
    "definitivo" (adaptado + re-QC) es un subconjunto distinto al de OE2
    (adaptado sin filtrar): el re-QC descarta algunos puntos adaptados.
    `capas` controla qué trazas se dibujan (para poder quitar/poner capas).

    `max_puntos` es deliberadamente alto (200 000) para que con el volumen
    de datos actual no haya sub-muestreo y el detalle coincida con la
    gráfica "Evolución: crudo vs. corregido" de OE2. Se conserva como
    salvaguarda para no saturar el navegador con conjuntos mucho mayores."""
    col_adapt = f"GHI_corregido_{variante}"
    col_usab = f"usabilidad_{variante}"

    # Referencia satelital (sombreada). Dato tal cual (sin promediar) —
    # solo se sub-muestrea con el mismo criterio que las demas capas para
    # no saturar el navegador, NUNCA se promedia ni se suaviza el valor.
    d_solcast = None
    if "solcast" in capas:
        df_sol = cargar_procesado(f"{SOLCAST_TAG_POR_FUENTE.get(nombre_clave, 'solcast')}_qc_completo.csv")
        if df_sol is not None and {"datetime", "GHI"}.issubset(df_sol.columns):
            d_solcast = df_sol.dropna(subset=["datetime", "GHI"]).copy()
            n_total_s = len(d_solcast)
            step_s = max(1, n_total_s // max_puntos)
            if step_s > 1:
                d_solcast = d_solcast.iloc[::step_s].copy()

    # Crudo + QC de OE1 (mismo criterio que fig_evolucion_qc_plotly)
    d_crudo, mask_crudo_ok = None, None
    if "crudo" in capas or "crudo_qc" in capas:
        df_crudo = cargar_procesado(f"{nombre_clave}_qc_completo.csv")
        if df_crudo is not None and {"datetime", "GHI"}.issubset(df_crudo.columns):
            d_crudo = df_crudo.dropna(subset=["datetime", "GHI"]).copy()
            n_total = len(d_crudo)
            step = max(1, n_total // max_puntos)
            if step > 1:
                d_crudo = d_crudo.iloc[::step].copy()
            flags = [c for c in d_crudo.columns if c.startswith("flag_")
                     and not any(diag in c for diag in _PRUEBAS_SOLO_DIAGNOSTICO)]
            m = pd.Series(True, index=d_crudo.index)
            for c in flags:
                m &= (d_crudo[c] != "FLAG")
            mask_crudo_ok = m

    # Adaptado (OE2) + re-QC (OE3)
    d_adapt, mask_adapt_ok = None, None
    if "adapt_100" in capas or "adapt_qc" in capas:
        df_adapt = cargar_procesado(f"{nombre_clave}_ghi_corregido.csv")
        if df_adapt is not None and {"datetime", col_adapt, col_usab}.issubset(df_adapt.columns):
            d_adapt = df_adapt.dropna(subset=["datetime", col_adapt]).copy()
            n_total_a = len(d_adapt)
            step_a = max(1, n_total_a // max_puntos)
            if step_a > 1:
                d_adapt = d_adapt.iloc[::step_a].copy()
            mask_adapt_ok = (d_adapt[col_usab] == "OK")

    if d_crudo is None and d_adapt is None and d_solcast is None:
        return None

    fig = go.Figure()

    # Paleta fija de alto contraste cromático (independiente de la
    # estación): rojo/verde para el par crudo -> crudo-que-pasa-QC, y
    # amarillo/azul para el par adaptado -> adaptado-que-pasa-re-QC.
    # Máxima diferencia de matiz dentro de cada par para que el
    # "antes/después" de cada etapa se note de inmediato.
    _COLOR_CRUDO = "#D7263D"        # rojo
    _COLOR_CRUDO_QC = "#2A9D57"     # verde
    _COLOR_ADAPT_100 = "#B8860B"    # amarillo oscuro (dorado)
    _COLOR_ADAPT_QC = "#1D6FB8"     # azul

    if d_solcast is not None and len(d_solcast) > 0:
        fig.add_trace(go.Scatter(
            x=d_solcast["datetime"], y=d_solcast["GHI"], mode="lines",
            line=dict(color="#999999", width=0.6),
            fill="tozeroy", fillcolor="rgba(120,120,120,0.15)",
            name="Solcast (sombra, dato horario tal cual)",
            hovertemplate="%{x|%d %b %Y %H:%M}<br>Solcast: %{y:.0f} W/m²<extra></extra>",
        ))

    if d_crudo is not None:
        pct_crudo_ok = mask_crudo_ok.mean() * 100 if mask_crudo_ok is not None and len(mask_crudo_ok) else 0
        if "crudo" in capas:
            fig.add_trace(go.Scatter(
                x=d_crudo["datetime"], y=d_crudo["GHI"], mode="lines",
                line=dict(color=_COLOR_CRUDO, width=1),
                opacity=0.65,
                name="Crudo (100%)",
                hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
            ))
        if "crudo_qc" in capas:
            fig.add_trace(go.Scatter(
                x=d_crudo.loc[mask_crudo_ok, "datetime"], y=d_crudo.loc[mask_crudo_ok, "GHI"], mode="lines",
                line=dict(color=_COLOR_CRUDO_QC, width=1.3),
                opacity=0.9,
                name=f"Crudo que pasa control de calidad (~{pct_crudo_ok:.0f}%)",
                hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
            ))

    if d_adapt is not None:
        pct_adapt_ok = mask_adapt_ok.mean() * 100 if mask_adapt_ok is not None and len(mask_adapt_ok) else 0
        if "adapt_100" in capas:
            fig.add_trace(go.Scatter(
                x=d_adapt["datetime"], y=d_adapt[col_adapt], mode="lines",
                line=dict(color=_COLOR_ADAPT_100, width=1.5),
                opacity=0.9,
                name=f"Adaptado — {variante} (100%, antes del re-QC)",
                hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
            ))
        if "adapt_qc" in capas:
            fig.add_trace(go.Scatter(
                x=d_adapt.loc[mask_adapt_ok, "datetime"], y=d_adapt.loc[mask_adapt_ok, col_adapt], mode="lines",
                line=dict(color=_COLOR_ADAPT_QC, width=1.9),
                opacity=0.95,
                name=f"Adaptado que pasa el re-control de calidad (~{pct_adapt_ok:.0f}%)",
                hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text=f"{nombre_display} — evolución completa: Solcast → crudo → control de "
                         f"calidad → adaptado ({variante}) → re-control de calidad", font=dict(size=13)),
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=65, b=20, l=60, r=190),
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            font=dict(size=10), bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1, itemsizing="constant",
        ),
        xaxis=dict(
            showgrid=False, tickformat="%b %Y",
            rangeselector=dict(buttons=[
                dict(count=7,  label="1S",  step="day",   stepmode="backward"),
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(step="all", label="Todo"),
            ], bgcolor="#f0f6f9", activecolor="#1F4E5F", font=dict(size=10)),
            rangeslider=dict(visible=True, thickness=0.07),
        ),
        yaxis=dict(title=dict(text="GHI [W/m²]", font=dict(size=11)), gridcolor="#eeeeee", rangemode="tozero"),
    )
    return fig

@st.cache_data(show_spinner=False)
def fig_envolvente_plotly(df: pd.DataFrame, col: str, nombre: str, colorscale=None,
                          bin_az: float = 2.0, bin_el: float = 1.0) -> "go.Figure | None":
    """Envolvente maxima: para cada celda (azimut, elevacion) se pinta el
    maximo valor de `col` observado en todo el periodo."""
    if df is None or not {col, "azimuth", "elevation"}.issubset(df.columns):
        return None
    if colorscale is None:
        colorscale = ESCALA_CALOR_SOLAR
    d = df.dropna(subset=[col, "azimuth", "elevation"]).copy()
    d = d[d["elevation"] > -5]
    if d.empty:
        return None
    d["az_bin"] = (d["azimuth"] // bin_az) * bin_az
    d["el_bin"] = (d["elevation"] // bin_el) * bin_el
    piv = d.groupby(["el_bin", "az_bin"])[col].max().unstack()

    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index, colorscale=colorscale,
        zmin=0, colorbar=dict(title=f"max {col} [W/m²]"),
        hovertemplate="Azimut %{x:.0f}°<br>Elevación %{y:.0f}°<br>Máx " + col + ": %{z:.0f} W/m²<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{nombre} — envolvente máxima de {col}", font=dict(size=13)),
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=40, b=20, l=40, r=10),
        xaxis=dict(title="Azimut solar [°]"),
        yaxis=dict(title="Elevación solar [°]"),
    )
    return fig

@st.cache_data(show_spinner=False)
def fig_envolvente_comparativo_plotly(dfs: tuple, etiquetas: tuple, col: str, nombre: str,
                                       colorscale=None, bin_az: float = 2.0,
                                       bin_el: float = 1.0, alto_por_fila: int = 190) -> "go.Figure | None":
    """Envolvente máxima de `col`, en UNA figura con N subplots apilados
    verticalmente (uno por etapa, cada uno a todo el ancho) compartiendo
    una sola escala de color y una sola barra de color, para que las
    versiones sean directamente comparables entre sí. Sirve tanto para 2
    etapas (OE1: crudo/post-QC) como para 3 (OE2: crudo/post-QC/site-adaptado)."""
    if colorscale is None:
        colorscale = ESCALA_CALOR_SOLAR

    def _pivot(df):
        """Matriz elevación x azimut (por bins) con el máximo de `col`; None si no hay datos."""
        if df is None or not {col, "azimuth", "elevation"}.issubset(df.columns):
            return None
        d = df.dropna(subset=[col, "azimuth", "elevation"]).copy()
        d = d[d["elevation"] > -5]
        if d.empty:
            return None
        d["az_bin"] = (d["azimuth"] // bin_az) * bin_az
        d["el_bin"] = (d["elevation"] // bin_el) * bin_el
        return d.groupby(["el_bin", "az_bin"])[col].max().unstack()

    pivots = [_pivot(df) for df in dfs]
    if all(p is None for p in pivots):
        return None

    zmax = max(
        (float(p.values[~pd.isna(p.values)].max()) for p in pivots
         if p is not None and p.size and (~pd.isna(p.values)).any()),
        default=None,
    )
    if zmax is None or not np.isfinite(zmax):
        return None

    n = len(pivots)
    fig = make_subplots(rows=n, cols=1, subplot_titles=etiquetas,
                         shared_xaxes=True, vertical_spacing=0.06)
    for i, piv in enumerate(pivots):
        if piv is None:
            continue
        fig.add_trace(go.Heatmap(
            z=piv.values, x=piv.columns, y=piv.index,
            coloraxis="coloraxis",
            hovertemplate="Azimut %{x:.0f}°<br>Elevación %{y:.0f}°<br>Máx " + col + ": %{z:.0f} W/m²<extra></extra>",
        ), row=i + 1, col=1)
        fig.update_yaxes(title_text="Elevación [°]", row=i + 1, col=1)
        if i == n - 1:
            fig.update_xaxes(title_text="Azimut [°]", showticklabels=True, row=i + 1, col=1)
        else:
            fig.update_xaxes(showticklabels=False, row=i + 1, col=1)

    fig.update_layout(
        title=dict(text=f"{nombre} — envolvente máxima de {col}",
                    font=dict(size=13)),
        height=alto_por_fila * n + 50, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=50, b=20, l=50, r=10),
        coloraxis=dict(colorscale=colorscale, cmin=0, cmax=zmax,
                        colorbar=dict(title=f"máx {col} [W/m²]", thickness=14)),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 12
    return fig

@st.cache_data(show_spinner=False)
def fig_simetria_ampm_azimut_plotly(df: pd.DataFrame, nombre: str, color: str,
                                    min_muestras_por_mitad: int = 10,
                                    tol_pct: float = 15.0) -> "go.Figure | None":
    """Asimetria diaria de energia AM vs PM (100*(E_am-E_pm)/(E_am+E_pm))
    contra el azimut solar de mediodia de ese dia. Un dia solo entra si
    ambas mitades tienen >= min_muestras_por_mitad lecturas validas de GHI
    y energia > 0 (evita que medios dias caidos den +-100% artificial)."""
    req = {"GHI", "elevation", "azimuth", "datetime"}
    if df is None or not req.issubset(df.columns):
        return None
    d = df[["datetime", "GHI", "elevation", "azimuth"]].copy()
    d["GHI"] = pd.to_numeric(d["GHI"], errors="coerce")
    d["fecha"] = d["datetime"].dt.date
    d["am"] = d["datetime"].dt.hour < 12
    d = d[d["elevation"] > 0]
    if d.empty:
        return None

    filas = []
    for fecha, grupo in d.groupby("fecha"):
        ghi_am = grupo.loc[grupo["am"], "GHI"]
        ghi_pm = grupo.loc[~grupo["am"], "GHI"]
        n_am, n_pm = ghi_am.notna().sum(), ghi_pm.notna().sum()
        am, pm = ghi_am.sum(), ghi_pm.sum()
        if n_am < min_muestras_por_mitad or n_pm < min_muestras_por_mitad or am <= 0 or pm <= 0:
            continue
        asimetria_pct = 100 * (am - pm) / (am + pm)
        azimut_mediodia = grupo.loc[grupo["elevation"].idxmax(), "azimuth"]
        filas.append({"fecha": fecha, "asimetria_pct": asimetria_pct, "azimut_mediodia": azimut_mediodia})

    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tabla["azimut_mediodia"], y=tabla["asimetria_pct"], mode="markers",
        marker=dict(color=color, size=6, opacity=0.55),
        name="Asimetría diaria",
        hovertemplate="Azimut: %{x:.0f}°<br>Asimetría: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=tol_pct, line_dash="dash", line_color="#E63946",
                  annotation_text=f"Umbral FLAG (±{tol_pct:.0f}%)", annotation_position="top right")
    fig.add_hline(y=-tol_pct, line_dash="dash", line_color="#E63946")
    fig.add_hline(y=0, line_color="#999", line_width=0.8)
    fig.update_layout(
        title=dict(text=f"{nombre} — simetría AM/PM vs azimut de mediodía", font=dict(size=13)),
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=40, b=20, l=40, r=10), showlegend=False,
        xaxis=dict(title="Azimut solar de mediodía [°] (0/360=Norte, 180=Sur)",
                    range=[0, 360], gridcolor="#f0f0f0"),
        yaxis=dict(title="Asimetría AM/PM [%] = 100·(E_AM−E_PM)/(E_AM+E_PM)", gridcolor="#f0f0f0"),
    )
    return fig

@st.cache_data(show_spinner=False)
def fig_simetria_comparativo_plotly(df_crudo: "pd.DataFrame | None", df_postqc: "pd.DataFrame | None",
                                     nombre: str, min_muestras_por_mitad: int = 10,
                                     tol_pct: float = 15.0) -> "go.Figure | None":
    """Asimetría AM/PM diaria vs. azimut de mediodía, crudo vs. post-QC, en
    UNA sola gráfica con 2 trazas de colores de alto contraste (C_RED /
    C_AZUL_VIVO) para comparar directamente el efecto del control de
    calidad inicial."""
    def _tabla(df):
        """Tabla diaria de asimetría AM/PM [%] y azimut de mediodía; None si no hay días válidos."""
        req = {"GHI", "elevation", "azimuth", "datetime"}
        if df is None or not req.issubset(df.columns):
            return None
        d = df[["datetime", "GHI", "elevation", "azimuth"]].copy()
        d["GHI"] = pd.to_numeric(d["GHI"], errors="coerce")
        d["fecha"] = d["datetime"].dt.date
        d["am"] = d["datetime"].dt.hour < 12
        d = d[d["elevation"] > 0]
        if d.empty:
            return None
        filas = []
        for fecha, grupo in d.groupby("fecha"):
            ghi_am = grupo.loc[grupo["am"], "GHI"]
            ghi_pm = grupo.loc[~grupo["am"], "GHI"]
            n_am, n_pm = ghi_am.notna().sum(), ghi_pm.notna().sum()
            am, pm = ghi_am.sum(), ghi_pm.sum()
            if n_am < min_muestras_por_mitad or n_pm < min_muestras_por_mitad or am <= 0 or pm <= 0:
                continue
            asimetria_pct = 100 * (am - pm) / (am + pm)
            azimut_mediodia = grupo.loc[grupo["elevation"].idxmax(), "azimuth"]
            filas.append({"asimetria_pct": asimetria_pct, "azimut_mediodia": azimut_mediodia})
        tabla = pd.DataFrame(filas)
        return tabla if not tabla.empty else None

    t_crudo, t_postqc = _tabla(df_crudo), _tabla(df_postqc)
    if t_crudo is None and t_postqc is None:
        return None

    fig = go.Figure()
    if t_crudo is not None:
        fig.add_trace(go.Scatter(
            x=t_crudo["azimut_mediodia"], y=t_crudo["asimetria_pct"], mode="markers",
            marker=dict(color=C_RED, size=6, opacity=0.5),
            name="Crudo",
            hovertemplate="Azimut: %{x:.0f}°<br>Asimetría (crudo): %{y:.1f}%<extra></extra>",
        ))
    if t_postqc is not None:
        fig.add_trace(go.Scatter(
            x=t_postqc["azimut_mediodia"], y=t_postqc["asimetria_pct"], mode="markers",
            marker=dict(color=C_AZUL_VIVO, size=6, opacity=0.6, symbol="diamond"),
            name="Post-QC",
            hovertemplate="Azimut: %{x:.0f}°<br>Asimetría (post-QC): %{y:.1f}%<extra></extra>",
        ))
    fig.add_hline(y=tol_pct, line_dash="dash", line_color="#999",
                  annotation_text=f"Umbral FLAG (±{tol_pct:.0f}%)", annotation_position="top right")
    fig.add_hline(y=-tol_pct, line_dash="dash", line_color="#999")
    fig.add_hline(y=0, line_color="#999", line_width=0.8)
    fig.update_layout(
        title=dict(text=f"{nombre} — simetría AM/PM vs azimut de mediodía", font=dict(size=13)),
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=45, b=20, l=40, r=10),
        legend=dict(orientation="h", y=1.1, x=0),
        xaxis=dict(title="Azimut solar de mediodía [°] (0/360=Norte, 180=Sur)",
                    range=[0, 360], gridcolor="#f0f0f0"),
        yaxis=dict(title="Asimetría AM/PM [%] = 100·(E_AM−E_PM)/(E_AM+E_PM)", gridcolor="#f0f0f0"),
    )
    return fig

@st.cache_data(show_spinner=False)
def fig_simetria_comparativo3_plotly(df_crudo: "pd.DataFrame | None", df_postqc: "pd.DataFrame | None",
                                      df_siteadapt: "pd.DataFrame | None", nombre: str,
                                      min_muestras_por_mitad: int = 10,
                                      min_muestras_por_mitad_sa: int = 10,
                                      tol_pct: float = 15.0) -> "go.Figure | None":
    """Asimetría AM/PM diaria vs. azimut de mediodía en UNA sola gráfica con
    3 trazas de alto contraste: crudo (rojo), post-QC (azul) y
    site-adaptado (verde) -- el resultado de OE2, agregado a las 2 etapas
    ya usadas en OE1.

    Las 3 etapas están a la misma resolución nativa de 5 min (todo el
    proyecto trabaja a esa resolución, no a promedios de 1h), así que
    usan el mismo umbral mínimo de muestras por medio día."""
    def _tabla(df, min_muestras):
        """Tabla diaria de asimetría AM/PM [%] y azimut de mediodía, exigiendo
        `min_muestras` lecturas válidas por medio día; None si no hay días válidos."""
        req = {"GHI", "elevation", "azimuth", "datetime"}
        if df is None or not req.issubset(df.columns):
            return None
        d = df[["datetime", "GHI", "elevation", "azimuth"]].copy()
        d["GHI"] = pd.to_numeric(d["GHI"], errors="coerce")
        d["fecha"] = d["datetime"].dt.date
        d["am"] = d["datetime"].dt.hour < 12
        d = d[d["elevation"] > 0]
        if d.empty:
            return None
        filas = []
        for fecha, grupo in d.groupby("fecha"):
            ghi_am = grupo.loc[grupo["am"], "GHI"]
            ghi_pm = grupo.loc[~grupo["am"], "GHI"]
            n_am, n_pm = ghi_am.notna().sum(), ghi_pm.notna().sum()
            am, pm = ghi_am.sum(), ghi_pm.sum()
            if n_am < min_muestras or n_pm < min_muestras or am <= 0 or pm <= 0:
                continue
            asimetria_pct = 100 * (am - pm) / (am + pm)
            azimut_mediodia = grupo.loc[grupo["elevation"].idxmax(), "azimuth"]
            filas.append({"asimetria_pct": asimetria_pct, "azimut_mediodia": azimut_mediodia})
        tabla = pd.DataFrame(filas)
        return tabla if not tabla.empty else None

    t_crudo = _tabla(df_crudo, min_muestras_por_mitad)
    t_postqc = _tabla(df_postqc, min_muestras_por_mitad)
    t_sa = _tabla(df_siteadapt, min_muestras_por_mitad_sa)
    if t_crudo is None and t_postqc is None and t_sa is None:
        return None

    fig = go.Figure()
    if t_crudo is not None:
        fig.add_trace(go.Scatter(
            x=t_crudo["azimut_mediodia"], y=t_crudo["asimetria_pct"], mode="markers",
            marker=dict(color=C_RED, size=6, opacity=0.45),
            name="① Crudo",
            hovertemplate="Azimut: %{x:.0f}°<br>Asimetría (crudo): %{y:.1f}%<extra></extra>",
        ))
    if t_postqc is not None:
        fig.add_trace(go.Scatter(
            x=t_postqc["azimut_mediodia"], y=t_postqc["asimetria_pct"], mode="markers",
            marker=dict(color=C_AZUL_VIVO, size=6, opacity=0.55, symbol="diamond"),
            name="② Post-QC",
            hovertemplate="Azimut: %{x:.0f}°<br>Asimetría (post-QC): %{y:.1f}%<extra></extra>",
        ))
    if t_sa is not None:
        fig.add_trace(go.Scatter(
            x=t_sa["azimut_mediodia"], y=t_sa["asimetria_pct"], mode="markers",
            marker=dict(color="#22AA44", size=6, opacity=0.65, symbol="square"),
            name="③ Site-adaptado",
            hovertemplate="Azimut: %{x:.0f}°<br>Asimetría (site-adaptado): %{y:.1f}%<extra></extra>",
        ))
    fig.add_hline(y=tol_pct, line_dash="dash", line_color="#999",
                  annotation_text=f"Umbral FLAG (±{tol_pct:.0f}%)", annotation_position="top right")
    fig.add_hline(y=-tol_pct, line_dash="dash", line_color="#999")
    fig.add_hline(y=0, line_color="#999", line_width=0.8)
    fig.update_layout(
        title=dict(text=f"{nombre} — simetría AM/PM", font=dict(size=13)),
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=45, b=20, l=40, r=10),
        legend=dict(orientation="h", y=1.1, x=0),
        xaxis=dict(title="Azimut solar de mediodía [°] (0/360=Norte, 180=Sur)",
                    range=[0, 360], gridcolor="#f0f0f0"),
        yaxis=dict(title="Asimetría AM/PM [%] = 100·(E_AM−E_PM)/(E_AM+E_PM)", gridcolor="#f0f0f0"),
    )
    return fig

_SIMBOLOS_ETAPAS_DIAG = ["circle", "diamond", "square", "x"]

@st.cache_data(show_spinner=False)
def fig_simetria_comparativo_n_plotly(dfs: tuple, etiquetas: tuple, nombre: str,
                                       min_muestras_por_mitad: int = 10,
                                       min_muestras_por_mitad_horario: int = 10,
                                       tol_pct: float = 15.0) -> "go.Figure | None":
    """Asimetría AM/PM diaria vs. azimut de mediodía en UNA sola gráfica con
    N trazas de alto contraste (una por etapa); generaliza
    fig_simetria_comparativo3_plotly a N etapas (p. ej. la cuarta,
    ④ post re-QC, en OE3).

    Todas las etapas están a la resolución nativa de 5 min, así que usan el
    mismo umbral mínimo de muestras por medio día:
    `min_muestras_por_mitad` aplica a las dos primeras etapas y
    `min_muestras_por_mitad_horario` a las siguientes (por defecto iguales)."""
    def _tabla(df, min_muestras):
        """Tabla diaria de asimetría AM/PM [%] y azimut de mediodía, exigiendo
        `min_muestras` lecturas válidas por medio día; None si no hay días válidos."""
        req = {"GHI", "elevation", "azimuth", "datetime"}
        if df is None or not req.issubset(df.columns):
            return None
        d = df[["datetime", "GHI", "elevation", "azimuth"]].copy()
        d["GHI"] = pd.to_numeric(d["GHI"], errors="coerce")
        d["fecha"] = d["datetime"].dt.date
        d["am"] = d["datetime"].dt.hour < 12
        d = d[d["elevation"] > 0]
        if d.empty:
            return None
        filas = []
        for fecha, grupo in d.groupby("fecha"):
            ghi_am = grupo.loc[grupo["am"], "GHI"]
            ghi_pm = grupo.loc[~grupo["am"], "GHI"]
            n_am, n_pm = ghi_am.notna().sum(), ghi_pm.notna().sum()
            am, pm = ghi_am.sum(), ghi_pm.sum()
            if n_am < min_muestras or n_pm < min_muestras or am <= 0 or pm <= 0:
                continue
            asimetria_pct = 100 * (am - pm) / (am + pm)
            azimut_mediodia = grupo.loc[grupo["elevation"].idxmax(), "azimuth"]
            filas.append({"asimetria_pct": asimetria_pct, "azimut_mediodia": azimut_mediodia})
        tabla = pd.DataFrame(filas)
        return tabla if not tabla.empty else None

    tablas = [
        _tabla(df, min_muestras_por_mitad if i < 2 else min_muestras_por_mitad_horario)
        for i, df in enumerate(dfs)
    ]
    if all(t is None for t in tablas):
        return None

    fig = go.Figure()
    for i, t in enumerate(tablas):
        if t is None:
            continue
        color = _COLORES_ETAPAS_DIAG[i % len(_COLORES_ETAPAS_DIAG)]
        simbolo = _SIMBOLOS_ETAPAS_DIAG[i % len(_SIMBOLOS_ETAPAS_DIAG)]
        fig.add_trace(go.Scatter(
            x=t["azimut_mediodia"], y=t["asimetria_pct"], mode="markers",
            marker=dict(color=color, size=6, opacity=0.45 + 0.07 * i, symbol=simbolo),
            name=etiquetas[i],
            hovertemplate="Azimut: %{x:.0f}°<br>Asimetría (" + etiquetas[i] + "): %{y:.1f}%<extra></extra>",
        ))
    fig.add_hline(y=tol_pct, line_dash="dash", line_color="#999",
                  annotation_text=f"Umbral FLAG (±{tol_pct:.0f}%)", annotation_position="top right")
    fig.add_hline(y=-tol_pct, line_dash="dash", line_color="#999")
    fig.add_hline(y=0, line_color="#999", line_width=0.8)
    fig.update_layout(
        title=dict(text=f"{nombre} — simetría AM/PM",
                    font=dict(size=13), y=0.97, yanchor="top"),
        height=440, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=80, b=20, l=40, r=10),
        legend=dict(orientation="h", y=1.16, x=0),
        xaxis=dict(title="Azimut solar de mediodía [°] (0/360=Norte, 180=Sur)",
                    range=[0, 360], gridcolor="#f0f0f0"),
        yaxis=dict(title="Asimetría AM/PM [%] = 100·(E_AM−E_PM)/(E_AM+E_PM)", gridcolor="#f0f0f0"),
    )
    return fig

@st.cache_data(show_spinner=False)
def fig_sesgo_dni_azimut_plotly(df: pd.DataFrame, nombre: str, color: str) -> "go.Figure | None":
    """Sesgo de DNI (medido - calculado por cierre) vs azimut solar.
    DNI_calc se despeja de GHI = DNI*cos(SZA) + DHI. Se traza el scatter
    de fondo (muestra) mas la mediana por bin de 10 grados, que es mas
    robusta que el scatter crudo cerca de salida/puesta de sol."""
    req = {"GHI", "DNI", "DHI", "zenith", "azimuth"}
    if df is None or not req.issubset(df.columns):
        return None
    d = df[["GHI", "DNI", "DHI", "zenith", "azimuth"]].copy()
    for c in ["GHI", "DNI", "DHI"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    cos_sza = np.clip(np.cos(np.radians(d["zenith"])), 0.05, None)
    d["dni_calc"] = (d["GHI"] - d["DHI"]) / cos_sza
    d["sesgo"] = d["DNI"] - d["dni_calc"]
    d = d[(d["GHI"] > 50) & d["sesgo"].between(-500, 500)]
    if d.empty:
        return None

    muestra = d.sample(n=min(len(d), 30000), random_state=1)
    d["bin_azimut"] = pd.cut(d["azimuth"], bins=np.arange(0, 361, 10))
    mediana_bin = d.groupby("bin_azimut", observed=True)["sesgo"].median()
    centros = [iv.mid for iv in mediana_bin.index]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=muestra["azimuth"], y=muestra["sesgo"], mode="markers",
        marker=dict(color=color, size=3, opacity=0.15), name="Datos (muestra de 30k)",
    ))
    fig.add_trace(go.Scatter(
        x=centros, y=mediana_bin.values, mode="lines",
        line=dict(color="black", width=2.5), name="Mediana por bin de 10°",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#E63946",
                  annotation_text="Sesgo = 0 (cierre perfecto)", annotation_position="bottom right")
    fig.update_layout(
        title=dict(text=f"{nombre} — sesgo DNI vs azimut (cierre radiativo)", font=dict(size=13)),
        height=440, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=40, b=20, l=40, r=10),
        legend=dict(orientation="h", y=1.06, x=0, itemsizing="constant"),
        xaxis=dict(title="Azimut solar [°] (90=Este, 180=Sur, 270=Oeste)",
                    range=[0, 360], gridcolor="#f0f0f0"),
        yaxis=dict(title="Sesgo DNI = medido − calculado [W/m²]", gridcolor="#f0f0f0"),
    )
    return fig

# Umbrales de config.yaml (qc_experto) usados en los K-tests
_QE_KT_MAX = 1.35
_QE_KN_FACTOR = 1100.0
_QE_KN_ALT_COEF = 0.03
_QE_K_MAX_SZA_BAJA = 1.05
_QE_K_MAX_SZA_ALTA = 1.10
_QE_CIERRE_SZA_CORTE = 75.0
_SITIO_ALTITUD_M = 2451  # altitud del sitio del caso de estudio (config.yaml: sitio.altitud_m)

@st.cache_data(show_spinner=False)
def fig_ktests_plotly(df: pd.DataFrame) -> "tuple[go.Figure, go.Figure] | None":
    """K-tests del procedimiento de QC experto en dos paneles: K=DHI/GHI vs
    Kt=GHI/(ETN·cosSZA) y Kn=DNI/ETN vs Kt, coloreados por flag_ktests.
    Solo aplica a fuentes con las tres componentes (GHI/DNI/DHI), como la
    satelital. Devuelve (fig_k, fig_kn) o None si faltan columnas."""
    req = {"GHI", "DNI", "DHI", "zenith", "dni_extra", "flag_ktests"}
    if df is None or not req.issubset(df.columns):
        return None
    d = df[["GHI", "DNI", "DHI", "zenith", "dni_extra", "flag_ktests"]].copy()
    for c in ["GHI", "DNI", "DHI"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    cos_sza = np.clip(np.cos(np.radians(d["zenith"])), 0.01, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["Kt"] = d["GHI"] / (d["dni_extra"] * cos_sza)
        d["K"] = d["DHI"] / d["GHI"]
        d["Kn"] = d["DNI"] / d["dni_extra"]
    d = d[(d["GHI"] > 50) & d["flag_ktests"].isin(["OK", "FLAG"])]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["Kt", "K", "Kn"])
    if d.empty:
        return None

    m = d.sample(n=min(len(d), 30000), random_state=1)
    colores = np.where(m["flag_ktests"] == "FLAG", "#E63946", "#9aa0a6")

    fig_k = go.Figure(go.Scatter(
        x=m["Kt"], y=m["K"], mode="markers",
        marker=dict(color=colores, size=4, opacity=0.45),
        hovertemplate="Kt: %{x:.2f}<br>K: %{y:.2f}<extra></extra>",
    ))
    fig_k.add_vline(x=_QE_KT_MAX, line_dash="dash", line_color="black")
    fig_k.add_hline(y=_QE_K_MAX_SZA_BAJA, line_dash="dash", line_color="#1f5fa3")
    fig_k.add_hline(y=_QE_K_MAX_SZA_ALTA, line_dash="dash", line_color="#8a1f1f")
    fig_k.update_layout(
        title=dict(text="K vs Kt — límites BSRN", font=dict(size=13)),
        height=420, plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        margin=dict(t=40, b=20, l=40, r=10),
        xaxis=dict(title="Kt = GHI / (ETN·cos SZA)", gridcolor="#f0f0f0"),
        yaxis=dict(title="K = DHI / GHI", gridcolor="#f0f0f0"),
    )

    etn_prom = df["dni_extra"].mean()
    limite_kn_aprox = (_QE_KN_FACTOR + _QE_KN_ALT_COEF * _SITIO_ALTITUD_M) / etn_prom
    fig_kn = go.Figure(go.Scatter(
        x=m["Kt"], y=m["Kn"], mode="markers",
        marker=dict(color=colores, size=4, opacity=0.45),
        hovertemplate="Kt: %{x:.2f}<br>Kn: %{y:.2f}<extra></extra>",
    ))
    fig_kn.add_vline(x=_QE_KT_MAX, line_dash="dash", line_color="black")
    fig_kn.add_hline(y=limite_kn_aprox, line_dash="dash", line_color="#1f5fa3",
                      annotation_text=f"Kn max aprox. = {limite_kn_aprox:.2f}", annotation_position="top left")
    fig_kn.update_layout(
        title=dict(text="Kn vs Kt — límites BSRN", font=dict(size=13)),
        height=420, plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        margin=dict(t=40, b=20, l=40, r=10),
        xaxis=dict(title="Kt = GHI / (ETN·cos SZA)", gridcolor="#f0f0f0"),
        yaxis=dict(title="Kn = DNI / ETN", gridcolor="#f0f0f0"),
    )
    return fig_k, fig_kn

def tooltip(texto: str):
    """Muestra un bloque de explicación desplegable, con etiqueta
    "Explicación" y fondo blanco homogéneo con el resto de la app."""
    with st.expander("Explicación", expanded=False):
        st.markdown(texto)

def kpi(col, valor, label, delta=None, delta_color="normal", formato="{:.1f}"):
    """Tarjeta de métrica (st.metric) en la columna `col`, con formato numérico opcional."""
    if isinstance(valor, float):
        valor_str = formato.format(valor)
    else:
        valor_str = str(valor)
    col.metric(label=label, value=valor_str, delta=delta, delta_color=delta_color)

def seccion(titulo: str, color: str = C_TEAL):
    """Encabezado de sección (estilo h2) con línea inferior del color indicado.
    Se usa <div>, no <h2>, para que Streamlit no le agregue el ícono de enlace
    automático (ancla) que no tiene ninguna función util en esta app de una
    sola página con navegación por barra lateral."""
    st.markdown(
        f"<div style='font-size:1.75rem; font-weight:600; color:{color}; "
        f"border-bottom:3px solid {color}; padding-bottom:6px; "
        f"margin-top:2rem'>{titulo}</div>",
        unsafe_allow_html=True,
    )

def subseccion(titulo: str, color: str = C_BLUE):
    """Encabezado de subsección (estilo h3), un nivel debajo de seccion().
    <div>, no <h3>, por la misma razón que en seccion()."""
    st.markdown(
        f"<div style='font-size:1.25rem; font-weight:600; color:{color}; "
        f"margin-top:1.5rem'>{titulo}</div>",
        unsafe_allow_html=True,
    )

def badge(texto: str, color: str = C_TEAL):
    """Etiqueta pequeña tipo 'pill' (fondo de color, texto blanco) para resaltar una palabra."""
    st.markdown(
        f"<span style='background:{color}; color:white; padding:3px 10px; "
        f"border-radius:12px; font-size:0.82rem; font-weight:600'>{texto}</span>",
        unsafe_allow_html=True,
    )

def panel_info(texto: str, color: str = C_LIGHT):
    """Bloque de explicación desplegable, idéntico en estilo a tooltip
    (etiqueta "Explicación", fondo blanco homogéneo), para notas o
    aclaraciones de una gráfica o tabla. `color` no se usa."""
    with st.expander("Explicación", expanded=False):
        st.markdown(texto)

def tabla_html(df: pd.DataFrame, formato: dict | None = None,
                negrilla_filas=None, negrilla_celdas: set | None = None,
                alto: int | None = None):
    """Tabla HTML con estilo consistente en toda la app, en lugar de
    st.dataframe(...): cabecera azul oscuro con letras blancas en negrilla y
    filas alternadas blanco / azul claro. Para resaltar un dato puntual se
    usa negrilla, no color de fondo.

    formato: dict opcional {columna: "{:.2f}%" o callable} -- igual que
        Styler.format(), se aplica antes de renderizar.
    negrilla_filas: lista/Series booleana (mismo largo que df) -- filas
        completas en negrilla (ej. la fila del modelo ganador).
    negrilla_celdas: set opcional de (indice_fila_posicional, columna) --
        celdas puntuales en negrilla.
    alto: si se da, la tabla queda con scroll vertical a esa altura en px
        (para tablas largas); si no, se muestra completa."""
    df = df.reset_index(drop=True)
    n = len(df)
    negrilla_filas = list(negrilla_filas) if negrilla_filas is not None else [False] * n
    negrilla_celdas = negrilla_celdas or set()

    def _fmt(col, val):
        """Texto de una celda según `formato` (vacío si es NaN)."""
        if pd.isna(val):
            return ""
        if formato and col in formato:
            f = formato[col]
            try:
                return f(val) if callable(f) else f.format(val)
            except Exception:
                return str(val)
        return str(val)

    filas_html = []
    for i in range(n):
        fila = df.iloc[i]
        bg = "#FFFFFF" if i % 2 == 0 else "#E8F1FA"
        celdas = []
        for col in df.columns:
            val_txt = _fmt(col, fila[col])
            es_negrilla = bool(negrilla_filas[i]) or (i, col) in negrilla_celdas
            peso = "font-weight:700;" if es_negrilla else ""
            celdas.append(
                f"<td style='padding:6px 10px; border-bottom:1px solid #e6e6e6; "
                f"white-space:nowrap; {peso}'>{val_txt}</td>"
            )
        filas_html.append(f"<tr style='background:{bg}'>{''.join(celdas)}</tr>")

    encabezado = "".join(
        f"<th style='background:{C_TEAL}; color:#ffffff; font-weight:700; "
        f"padding:8px 10px; text-align:left; white-space:nowrap; "
        f"position:sticky; top:0'>{c}</th>"
        for c in df.columns
    )
    contenedor_estilo = "overflow-x:auto; max-width:100%; border-radius:8px; border:1px solid #e6e6e6;"
    if alto:
        contenedor_estilo += f" max-height:{alto}px; overflow-y:auto;"
    st.markdown(
        f"<div style='{contenedor_estilo}'>"
        f"<table style='border-collapse:collapse; width:100%; font-size:0.88rem'>"
        f"<thead><tr>{encabezado}</tr></thead>"
        f"<tbody>{''.join(filas_html)}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )

def tabs_perezosas(etiquetas: list[str], key: str):
    """Reemplazo de st.tabs que SÍ carga de forma perezosa.

    st.tabs() ejecuta y monta en el DOM el contenido de TODAS las pestañas
    en cada rerun, aunque el usuario solo vea una (la selección es solo
    visual). Con decenas de gráficos Plotly y tablas por página, eso vuelve
    lenta la interfaz (scroll trabado, demora al pintar).

    Esta función usa un st.radio horizontal: cada pestaña es un bloque
    if/elif de Python, así que SOLO se ejecuta el bloque de la pestaña
    elegida; el resto ni se calcula ni se envía al navegador.

    Uso (en lugar de `with tabs[i]:` se escribe `if sel == "...":`):
        sel = tabs_perezosas(["Uno", "Dos"], key="mis_tabs")
        if sel == "Uno":
            ...
        elif sel == "Dos":
            ...
    """
    return st.radio(
        key, etiquetas, horizontal=True, label_visibility="collapsed", key=key,
    )


def render_mermaid(codigo: str, alto: int = 560):
    """Renderiza un diagrama Mermaid (sintaxis de texto: cajas + flechas)
    dentro de la app, cargando la librería mermaid.js desde CDN. Se usa
    Mermaid en lugar de Graphviz porque no requiere binarios del sistema
    (Graphviz necesita el ejecutable 'dot'), y en lugar de Plotly/matplotlib
    porque describir cajas y flechas en texto es mucho más simple.
    Se usa para los diagramas de flujo de OE1/OE2/OE3 y sirve para
    cualquier otro diagrama Mermaid. `alto` es la altura del panel en px.
    """
    html = f"""
    <style>
        .mermaid-scroll {{
            width: 100%;
            height: {alto}px;
            overflow: auto;
            border: 1px solid #ddeaf0;
            border-radius: 8px;
            padding: 10px;
            box-sizing: border-box;
            background: #ffffff;
        }}
    </style>
    <div class="mermaid-scroll">
        <div class="mermaid">
{codigo}
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: "base",
            themeVariables: {{
                fontFamily: "sans-serif",
                primaryTextColor: "#1a1a1a",
                lineColor: "#888888"
            }},
            flowchart: {{
                curve: "basis",
                htmlLabels: true,
                useMaxWidth: false
            }}
        }});
    </script>
    """
    components.html(html, height=alto + 20, scrolling=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QC & Site-Adaptation | Recurso Solar Pasto",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS global
st.markdown(f"""
<style>
  /* Forzar tema claro sin importar el modo oscuro del sistema/navegador
     (el CSS de esta app da por hecho fondo blanco y texto oscuro en
     muchos bloques; con tema oscuro el fondo se vuelve negro y ese texto
     oscuro queda ilegible). Ver también .streamlit/config.toml. */
  [data-testid="stAppViewContainer"], [data-testid="stHeader"],
  .main, body {{
      background-color: #ffffff !important;
      color: #1a1a1a !important;
  }}
  [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] span,
  [data-testid="stAppViewContainer"] li, [data-testid="stAppViewContainer"] label,
  [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
  [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4 {{
      color: #1a1a1a;
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background: {C_TEAL};
  }}
  [data-testid="stSidebar"] * {{
      color: white !important;
  }}
  [data-testid="stSidebar"] .stRadio label {{
      font-size: 1rem;
      padding: 6px 0;
  }}

  /* Métricas */
  [data-testid="metric-container"] {{
      background: {C_LIGHT};
      border: 1px solid #ddeaf0;
      border-radius: 10px;
      padding: 14px 18px;
  }}
  [data-testid="stMetricValue"] {{
      color: {C_TEAL} !important;
      font-size: 1.6rem !important;
      font-weight: 700 !important;
  }}
  [data-testid="stMetricLabel"] {{
      color: #444 !important;
      font-size: 0.82rem !important;
  }}

  /* Tablas */
  [data-testid="stDataFrame"] {{
      border-radius: 8px;
      overflow: hidden;
  }}

  /* Expanders */
  [data-testid="stExpander"] {{
      border: 1px solid #cde0e8 !important;
      border-radius: 8px !important;
  }}

  /* Separadores */
  hr {{ border-color: {C_LIGHT}; }}

  /* Tabs */
  .stTabs [data-baseweb="tab"] {{
      font-size: 1rem;
      font-weight: 600;
      color: {C_TEAL};
  }}
  .stTabs [aria-selected="true"] {{
      border-bottom: 3px solid {C_TEAL} !important;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ Recurso Solar\n**Pasto, Nariño — 2025**")
    st.markdown("---")
    pagina = st.radio(
        "Navegar",
        ["🏠  Inicio", "🔬  OE1 — Control de Calidad", "📐  OE2 — Site-Adaptation", "📊  OE3 — Desempeño",
         "🌍  Nuevo dataset (demo)"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>Trabajo de grado — Maestría en Ingeniería Electrónica<br>"
        "Universidad de Nariño · 2025</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PÁGINA: INICIO  ██████████████████
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# DATOS CRUDOS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
COLORES_FUENTES = {
    # Homogéneo en toda la app: fuente satelital = azul, fuente terrestre
    # = verde; cada caso de estudio con su propio tono.
    "Solcast":              "#154360",   # azul oscuro (caso 2023)
    "Udenar (terrestre)":   "#145A32",   # verde oscuro (caso 2023)
    "Solcast 2025":         "#1F618D",   # azul oscuro, otro tono (caso 2025)
    NOMBRE_UDENAR_EST:      "#0E6251",   # verde oscuro, otro tono (caso 2025)
}

@st.cache_data(show_spinner="Cargando datos crudos...")
def cargar_crudos_global():
    """Carga el GHI crudo (sin QC) de las fuentes satelital y terrestre de
    los dos casos de estudio para el panel 'Datos crudos' de Inicio. Busca
    cada archivo primero en data/raw/ y luego en la raíz del proyecto.
    Devuelve {nombre_fuente: Serie GHI}."""
    DIR_RAW = ROOT / "data" / "raw"
    def _s(df, cdt, cghi, fmt=None):
        """Normaliza una fuente cruda: parsea su columna de fecha y devuelve la
        serie GHI indexada por tiempo y ordenada."""
        d = df.copy()
        d["_dt"] = pd.to_datetime(d[cdt], format=fmt, errors="coerce") if fmt                    else pd.to_datetime(d[cdt], errors="coerce")
        return d.dropna(subset=["_dt"]).set_index("_dt")[cghi].sort_index()
    fuentes = {}
    for r in [DIR_RAW/"solcast2023PT5M_Udenar.csv", ROOT/"solcast2023PT5M_Udenar.csv"]:
        if r.exists():
            df = pd.read_csv(r)
            df["_dt"] = (pd.to_datetime(df["period_end"], utc=True)
                         .dt.tz_convert("America/Bogota").dt.tz_localize(None))
            fuentes["Solcast"] = df.dropna(subset=["_dt"]).set_index("_dt")["ghi"].sort_index()
            break
    for f in [DIR_RAW/"udenar.csv", ROOT/"udenar.csv"]:
        if f.exists():
            fuentes["Udenar (terrestre)"] = _s(pd.read_csv(f), "date", "ghi"); break
    for r in [DIR_RAW/"solcast2025PT5M_Udenar.csv", ROOT/"solcast2025PT5M_Udenar.csv"]:
        if r.exists():
            df = pd.read_csv(r)
            df["_dt"] = (pd.to_datetime(df["period_end"], utc=True)
                         .dt.tz_convert("America/Bogota").dt.tz_localize(None))
            fuentes["Solcast 2025"] = df.dropna(subset=["_dt"]).set_index("_dt")["ghi"].sort_index()
            break
    for f in [DIR_RAW/"udenar_est.csv", ROOT/"udenar_est.csv"]:
        if f.exists():
            fuentes[NOMBRE_UDENAR_EST] = _s(pd.read_csv(f), "date", "irradiance", fmt="%m/%d/%Y %H:%M")
            break
    return fuentes


# ── Helpers de OE2 (reconstrucción de estaciones con la fuente satelital como referencia) ──

@st.cache_data(show_spinner=False)
def cargar_reconstruccion_tablas():
    """Tablas comunes de la reconstrucción (resumen de emparejamiento, métricas
    de modelos y comparativo entre estaciones); cada entrada es None si no existe."""
    return {
        "resumen": cargar_csv("reconstruccion_resumen_emparejamiento.csv"),
        "metricas": cargar_csv("reconstruccion_metricas_modelos.csv"),
        "comparativo": cargar_csv("reconstruccion_comparativo_4estaciones.csv"),
    }


@st.cache_data(show_spinner=False)
def fig_dispersion_1a1_plotly(clave: str, nombre: str):
    """Dispersión 1:1 GHI predicho vs. medido, antes (referencia satelital
    sin corregir) y después (modelo ganador de site-adaptation). Se usa en
    OE2 y en el mosaico de OE3. Devuelve (fig, stats), con stats un dict de
    pendiente/ángulo antes y después, o (None, None) si no hay datos."""
    tablas = cargar_reconstruccion_tablas()
    df_metricas = tablas["metricas"]
    _df_pred = cargar_procesado(f"reconstruccion_{clave}_predicciones_test.csv")
    col_real = f"GHI_{clave}"
    if (_df_pred is None or df_metricas is None or col_real not in _df_pred.columns
            or "Baseline (Solcast sin corregir)" not in _df_pred.columns):
        return None, None
    _dfm_g = df_metricas[(df_metricas["estacion"] == clave) & (df_metricas["modelo_ganador"])]
    if _dfm_g.empty:
        return None, None
    _ganador_col = _dfm_g.iloc[0]["modelo"]
    if _ganador_col not in _df_pred.columns:
        return None, None
    _dfp = _df_pred.dropna(subset=[col_real, "Baseline (Solcast sin corregir)", _ganador_col])
    if _dfp.empty:
        return None, None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_dfp[col_real], y=_dfp["Baseline (Solcast sin corregir)"],
        mode="markers", name="Antes — Solcast sin corregir",
        marker=dict(color=C_RED, size=5, opacity=0.45)))
    fig.add_trace(go.Scatter(
        x=_dfp[col_real], y=_dfp[_ganador_col],
        mode="markers", name=f"Después — modelo site-adaptado ({_ganador_col})",
        marker=dict(color=C_AZUL_VIVO, size=5, opacity=0.45)))
    _max_ax = float(np.nanmax([
        _dfp[col_real].max(), _dfp["Baseline (Solcast sin corregir)"].max(), _dfp[_ganador_col].max(),
    ])) * 1.05
    fig.add_trace(go.Scatter(
        x=[0, _max_ax], y=[0, _max_ax], mode="lines", name="Referencia 1:1",
        line=dict(color="#999", width=1, dash="dash")))

    _m_antes, _b_antes = np.polyfit(_dfp[col_real], _dfp["Baseline (Solcast sin corregir)"], 1)
    _m_desp, _b_desp = np.polyfit(_dfp[col_real], _dfp[_ganador_col], 1)
    _ang_antes = np.degrees(np.arctan(_m_antes))
    _ang_desp = np.degrees(np.arctan(_m_desp))
    _x_fit = np.array([0.0, _max_ax])
    fig.add_trace(go.Scatter(
        x=_x_fit, y=_m_antes * _x_fit + _b_antes, mode="lines",
        name=f"Ajuste antes (pendiente={_m_antes:.2f}, {_ang_antes:.0f}°)",
        line=dict(color=C_RED, width=2, dash="dot")))
    fig.add_trace(go.Scatter(
        x=_x_fit, y=_m_desp * _x_fit + _b_desp, mode="lines",
        name=f"Ajuste después (pendiente={_m_desp:.2f}, {_ang_desp:.0f}°)",
        line=dict(color=C_AZUL_VIVO, width=2, dash="dot")))

    fig.update_layout(
        title=dict(text="Dispersión 1:1 — GHI predicho vs. medido", font=dict(size=13)),
        height=560, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, font=dict(size=10)),
        margin=dict(t=40, b=100, l=10, r=10),
        xaxis=dict(title="GHI medido, post-QC [W/m²]", gridcolor="#eee"),
        yaxis=dict(title="GHI predicho [W/m²]", gridcolor="#eee", scaleanchor="x", scaleratio=1),
    )
    stats = dict(m_antes=_m_antes, ang_antes=_ang_antes, m_desp=_m_desp, ang_desp=_ang_desp,
                 ganador_col=_ganador_col)
    return fig, stats


@st.cache_data(show_spinner=False)
def fig_ecdf_plotly(clave: str, nombre: str) -> "go.Figure | None":
    """ECDF del GHI medido vs. referencia satelital sin corregir vs.
    site-adaptado (modelo ganador), en el conjunto de prueba. Se usa en OE2
    y en el mosaico de OE3. Devuelve None si no hay datos."""
    tablas = cargar_reconstruccion_tablas()
    df_metricas = tablas["metricas"]
    _df_pred = cargar_procesado(f"reconstruccion_{clave}_predicciones_test.csv")
    col_real = f"GHI_{clave}"
    if (_df_pred is None or df_metricas is None or col_real not in _df_pred.columns
            or "Baseline (Solcast sin corregir)" not in _df_pred.columns):
        return None
    _dfm_g = df_metricas[(df_metricas["estacion"] == clave) & (df_metricas["modelo_ganador"])]
    if _dfm_g.empty:
        return None
    _ganador_col = _dfm_g.iloc[0]["modelo"]
    if _ganador_col not in _df_pred.columns:
        return None
    _dfp = _df_pred.dropna(subset=[col_real, "Baseline (Solcast sin corregir)", _ganador_col])
    if _dfp.empty:
        return None

    def _ecdf(valores):
        """Función de distribución empírica: (valores ordenados, probabilidad acumulada)."""
        x = np.sort(np.asarray(valores, dtype=float))
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    _x_real, _y_real = _ecdf(_dfp[col_real])
    _x_crudo, _y_crudo = _ecdf(_dfp["Baseline (Solcast sin corregir)"])
    _x_adapt, _y_adapt = _ecdf(_dfp[_ganador_col])
    _x_max_ecdf = float(np.nanpercentile(
        np.concatenate([_x_real, _x_crudo, _x_adapt]), 99.5))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_x_real, y=_y_real, mode="lines", name="Medido (post-QC)",
        line=dict(color="#1a1a1a", width=2.5)))
    fig.add_trace(go.Scatter(x=_x_crudo, y=_y_crudo, mode="lines",
        name="Solcast", line=dict(color=C_RED, width=1.8, dash="dash")))
    fig.add_trace(go.Scatter(x=_x_adapt, y=_y_adapt, mode="lines",
        name=f"Adaptado ({_ganador_col})", line=dict(color=C_AZUL_VIVO, width=2)))
    fig.update_layout(
        title=dict(text="ECDF — medido vs. Solcast vs. site-adaptado", font=dict(size=13)),
        height=480, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        margin=dict(t=40, b=90, l=50, r=20),
        xaxis=dict(title="GHI [W/m²]", gridcolor="#eee", range=[0, _x_max_ecdf]),
        yaxis=dict(title="ECDF (-)", gridcolor="#eee", range=[0.4, 1.0]),
    )
    return fig


def render_tab_reconstruccion(clave: str, nombre_display: str, color: str):
    """Contenido de una pestaña de estación en OE2: KPIs, ranking de modelos,
    importancia de variables, serie corregida (2 variantes) y comparación
    de variantes tras re-aplicar el QC."""
    tablas = cargar_reconstruccion_tablas()
    df_resumen, df_metricas, df_comp = tablas["resumen"], tablas["metricas"], tablas["comparativo"]

    seccion(f"{nombre_display} — reconstrucción con Solcast como referencia", color)

    fila = None
    if df_resumen is not None and clave in df_resumen["estacion"].values:
        fila = df_resumen[df_resumen["estacion"] == clave].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Muestras supervisadas", f"{fila['muestras_emparejadas_supervisado']:,.0f}")
        c2.metric("Muestras de aplicación (año completo)", f"{fila['muestras_aplicacion_total']:,.0f}")
        pct_medido = 100 * fila["muestras_medido_ok"] / fila["muestras_aplicacion_total"] if fila["muestras_aplicacion_total"] else float("nan")
        c3.metric("% ya medido y válido", f"{pct_medido:.1f}%")
        c4.metric("Sesgo Solcast−estación", f"{fila['sesgo_medio_antes (Solcast-estacion, W/m2)']:+.1f} W/m²")
        panel_info(f"Correlación cruda Solcast↔{nombre_display}: {fila['correlacion_antes']:.3f} · "
                   f"Muestras train: {fila['n_train']:,.0f} · Muestras test: {fila['n_test']:,.0f}")
    else:
        st.info(f"No hay tabla de resumen para {nombre_display}. Corre el paso de emparejamiento "
                f"entre la fuente satelital y esta estación.")

    _ganador = None
    if df_metricas is not None:
        _dfm_est = df_metricas[df_metricas["estacion"] == clave].copy()
        if not _dfm_est.empty:
            subseccion("Ranking de modelos (conjunto de prueba)")
            if _dfm_est["modelo_ganador"].any():
                _ganador = _dfm_est.loc[_dfm_est["modelo_ganador"], "modelo"].iloc[0]

            _tiene_pct_inval = "pct_fisicamente_invalido" in _dfm_est.columns
            cols_show = ["ranking", "modelo", "RMSE", "MAE", "R2"]
            if _tiene_pct_inval:
                cols_show.append("pct_fisicamente_invalido")
            cols_show += ["n_bajo_minimo_fisico", "fisicamente_valido"]
            df_show = _dfm_est[cols_show].copy()
            nombres_cols = ["#", "Modelo", "RMSE [W/m²]", "MAE [W/m²]", "R²"]
            if _tiene_pct_inval:
                nombres_cols.append("% inválido (año completo)")
            nombres_cols += ["Predict. < mín. (test)", "Físic. válido"]
            df_show.columns = nombres_cols
            df_show["RMSE [W/m²]"] = df_show["RMSE [W/m²]"].round(2)
            df_show["MAE [W/m²]"] = df_show["MAE [W/m²]"].round(2)
            df_show["R²"] = df_show["R²"].round(4)
            if _tiene_pct_inval:
                df_show["% inválido (año completo)"] = df_show["% inválido (año completo)"].round(3)

            # Filas de referencia (crudo / post-QC vs. referencia satelital) en la MISMA
            # ventana de fechas del conjunto de prueba, para poder comparar
            # los 3 escalones -- crudo, post-QC, site-adaptado -- en pie de
            # igualdad. Ver `calcular_metricas_referencia_ventana()`.
            _nombres_ref = {"crudo": "① Crudo (sin QC) vs. Solcast",
                            "post_qc": "② Post-QC vs. Solcast"}
            filas_ref = []
            if fila is not None and str(fila.get("rango_test", "")).strip():
                _rango = _parsear_rango_test(fila["rango_test"])
                _ref = (calcular_metricas_referencia_ventana(clave, _rango[0], _rango[1])
                        if _rango is not None else None)
                if _ref is not None:
                    for _etq, _label in _nombres_ref.items():
                        _m = _ref.get(_etq)
                        if _m is None or pd.isna(_m.get("MAE")):
                            continue
                        _fila_ref = {
                            "#": "—", "Modelo": _label,
                            "RMSE [W/m²]": round(_m["RMSE"], 2), "MAE [W/m²]": round(_m["MAE"], 2),
                            "R²": round(_m["R2"], 4),
                            "Predict. < mín. (test)": "—", "Físic. válido": "—",
                        }
                        if _tiene_pct_inval:
                            _fila_ref["% inválido (año completo)"] = "—"
                        filas_ref.append(_fila_ref)
            if filas_ref:
                df_show = pd.concat([pd.DataFrame(filas_ref), df_show], ignore_index=True)

            # Fila del modelo ganador en negrilla; sin fondos de color para
            # las filas de referencia ni las excluidas/inválidas.
            _negrilla_filas = (df_show["Modelo"] == _ganador).tolist()
            tabla_html(df_show, negrilla_filas=_negrilla_filas)
            boton_descarga_csv(df_show, f"oe2_{clave}_ranking_modelos.csv")
            if filas_ref:
                panel_info(
                    "① y ② son puntos de referencia, no modelos candidatos — muestran cómo le "
                    "iba a la estación frente a Solcast ANTES de aplicar cualquier modelo de "
                    "site-adaptation, calculados en las mismas fechas del conjunto de prueba de "
                    "abajo (para que los 3 escalones sean comparables). El resto de la tabla es "
                    "'③ site-adaptado': cada modelo entrenado para predecir el GHI real de la "
                    "estación. Si ①→②→ganador va bajando el RMSE, es evidencia de que el QC y "
                    "luego el site-adaptation van mejorando la calidad del dato, en ese orden."
                    + (
                        " La columna '% inválido (año completo)' es el criterio real que decide "
                        "'Físic. válido': revisa las predicciones de cada modelo sobre TODO el "
                        "período de aplicación (no solo el conjunto de prueba), con las mismas "
                        "pruebas físicas BSRN (negativo, límites PPL/ERL, nocturno) que el resto "
                        "del proyecto. Un modelo puede salir limpio en 'Predict. < mín. (test)' y "
                        "aun así ser descartado aquí si extrapola mal en el resto del año."
                        if _tiene_pct_inval else ""
                    )
                )

            _dfm_sorted = _dfm_est.sort_values("RMSE", ascending=False)
            _colores_rk = [
                color if m == _ganador else "#cccccc" if not fv else "#a8d5e2"
                for m, fv in zip(_dfm_sorted["modelo"], _dfm_sorted["fisicamente_valido"])
            ]
            _fig_rk = go.Figure(go.Bar(
                x=_dfm_sorted["RMSE"], y=_dfm_sorted["modelo"], orientation="h",
                marker_color=_colores_rk,
                text=[f"{v:.1f}" for v in _dfm_sorted["RMSE"]], textposition="outside",
                hovertemplate="%{y}<br>RMSE: %{x:.2f} W/m²<extra></extra>",
            ))
            _fig_rk.update_layout(
                title=f"{nombre_display} — RMSE por modelo (test). Ganador: {_ganador}",
                height=360, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=50, b=20, l=10, r=40),
                xaxis=dict(title="RMSE [W/m²]", gridcolor="#f0f0f0"), yaxis=dict(title=""),
            )
            st.plotly_chart(_fig_rk, use_container_width=True, key=f"ranking_{clave}")
            panel_info(
                "Cada barra es un modelo, ordenado por RMSE en el conjunto de prueba (menor "
                "es mejor). Color destacado = modelo ganador; gris = descartado (no físicamente "
                "válido); celeste = válido pero no ganador."
            )

    _df_coef = cargar_csv(f"reconstruccion_{clave}_coeficientes.csv")
    if (_df_coef is not None and not _df_coef.empty and _ganador
            and str(_df_coef["modelo"].iloc[0]) == str(_ganador)):
        subseccion(f"Coeficientes del modelo ganador ({_ganador})")
        _b0 = float(_df_coef.loc[_df_coef["termino"] == "intercepto", "valor"].iloc[0])
        _resto = _df_coef[_df_coef["termino"] != "intercepto"]
        if len(_resto) == 1:
            _b1 = float(_resto["valor"].iloc[0])
            st.latex(rf"GHI_{{\text{{{nombre_display}}}}} = {_b0:.3f} + {_b1:.4f}\cdot GHI_{{\text{{Solcast}}}}")
            panel_info(
                f"El modelo ganador es una recta: por cada W/m² que marca Solcast, la "
                f"corrección asigna {_b1:.3f} W/m² (corrección de escala, pendiente menor que 1 "
                f"porque Solcast sobrestima la medida en tierra), y el intercepto {_b0:.2f} W/m² "
                "es el valor que predice cuando Solcast marca cero (noche)."
            )
        else:
            tabla_html(_df_coef[["termino", "valor"]])

    _df_imp = cargar_csv(f"reconstruccion_{clave}_importancia_variables.csv")
    if _df_imp is not None and not _df_imp.empty:
        subseccion(f"Importancia de variables ({_ganador or 'modelo ganador'})")
        _dfi = _df_imp.sort_values("importancia", ascending=True)
        _fig_imp = go.Figure(go.Bar(
            x=_dfi["importancia"] * 100, y=_dfi["variable"], orientation="h",
            marker_color=color,
            text=[f"{v:.1f}%" for v in _dfi["importancia"] * 100], textposition="outside",
            hovertemplate="%{y}<br>Importancia: %{x:.2f}%<extra></extra>",
        ))
        _fig_imp.update_layout(
            title=f"{nombre_display} — importancia de variables", height=340,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=50, b=20, l=10, r=40),
            xaxis=dict(title="Importancia [%]", gridcolor="#f0f0f0"), yaxis=dict(title=""),
        )
        st.plotly_chart(_fig_imp, use_container_width=True, key=f"importancia_{clave}")
        panel_info(
            "Qué tanto pesa cada variable predictora en las decisiones del modelo ganador "
            "(mayor % = más influyente). Ayuda a verificar que el modelo aprende relaciones "
            "físicamente razonables (ej. que el GHI satelital sea la variable más importante)."
        )

    _df_serie = cargar_procesado(f"{clave}_ghi_corregido.csv")
    if _df_serie is not None:
        subseccion("Evolución: crudo vs. corregido (2 variantes superpuestas)")
        _fig_serie = go.Figure()

        # Referencia satelital como sombra de fondo (dato tal cual), con el
        # mismo tratamiento visual que fig_evolucion_reconstruccion_plotly en
        # OE3, para mostrar el contexto satelital detrás de la corrección.
        _df_sol_fondo = cargar_procesado(f"{SOLCAST_TAG_POR_FUENTE.get(clave, 'solcast')}_qc_completo.csv")
        if _df_sol_fondo is not None and {"datetime", "GHI"}.issubset(_df_sol_fondo.columns):
            _fig_serie.add_trace(go.Scatter(
                x=_df_sol_fondo["datetime"], y=_df_sol_fondo["GHI"], mode="lines",
                line=dict(color="#999999", width=0.6),
                fill="tozeroy", fillcolor="rgba(120,120,120,0.15)",
                name="Solcast (sombra, dato horario tal cual)",
                hovertemplate="%{x|%d %b %Y %H:%M}<br>Solcast: %{y:.0f} W/m²<extra></extra>",
            ))

        _fig_serie.add_trace(go.Scatter(
            x=_df_serie["datetime"], y=_df_serie["GHI_original"], mode="lines",
            line=dict(color="#B02A2A", width=1.1),
            opacity=0.75,
            name="Crudo (medido)",
            hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
        ))
        _fig_serie.add_trace(go.Scatter(
            x=_df_serie["datetime"], y=_df_serie["GHI_corregido_total"], mode="lines",
            line=dict(color=COLOR_CORREGIDO_TOTAL, width=1.6),
            opacity=0.9,
            name="Corregido — total",
            hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
        ))
        _fig_serie.add_trace(go.Scatter(
            x=_df_serie["datetime"], y=_df_serie["GHI_corregido_selectivo"], mode="lines",
            line=dict(color=COLOR_CORREGIDO_SELECTIVO, width=1.2),
            opacity=0.9,
            name="Corregido — selectivo",
            hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
        ))
        _fig_serie.update_layout(
            title=f"{nombre_display} — GHI: crudo vs. corregido (total / selectivo)",
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40, b=20, l=60, r=170),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                        font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="#ccc", borderwidth=1, itemsizing="constant"),
            xaxis=dict(showgrid=False, tickformat="%b %Y",
                       rangeselector=dict(buttons=[
                           dict(count=7, label="1S", step="day", stepmode="backward"),
                           dict(count=1, label="1M", step="month", stepmode="backward"),
                           dict(count=3, label="3M", step="month", stepmode="backward"),
                           dict(step="all", label="Todo"),
                       ], bgcolor="#f0f6f9", activecolor="#1F4E5F", font=dict(size=10)),
                       rangeslider=dict(visible=True, thickness=0.07)),
            yaxis=dict(title=dict(text="GHI [W/m²]", font=dict(size=11)), gridcolor="#eeeeee", rangemode="tozero"),
        )
        st.plotly_chart(_fig_serie, use_container_width=True, key=f"serie_corregida_{clave}")
        panel_info(
            "Serie temporal completa: el dato crudo medido (rojo) contra las 2 variantes de "
            "corrección — total (reemplaza toda la serie por la predicción del modelo) y "
            "selectiva (conserva la medición cuando ya era válida, corrige solo lo demás). "
            "Usa el selector de rango para hacer zoom a un periodo específico."
        )
    else:
        st.info(f"No hay serie corregida para {nombre_display}. Corre el paso de reconstrucción "
                f"(aplicar el modelo ganador para corregir y rellenar la serie).")

    if df_comp is not None:
        _dfc = df_comp[df_comp["estacion"] == clave]
        if not _dfc.empty:
            subseccion("Comparación de variantes (después de re-aplicar el QC completo)")
            _dfc_show = _dfc[["variante", "modelo_ganador", "RMSE_test", "pct_usable_crudo",
                               "pct_usable_corregido", "mejora_pct_usable_pp",
                               "pct_medido_ok", "pct_medido_flag_corregido", "pct_sin_dato_rellenado"]].copy()
            _dfc_show.columns = ["Variante", "Modelo", "RMSE test [W/m²]", "% usable crudo",
                                  "% usable corregido", "Mejora [pp]", "% medido OK",
                                  "% corregido (flag)", "% rellenado (hueco)"]
            tabla_html(_dfc_show)
            boton_descarga_csv(_dfc_show, f"oe2_{clave}_comparativo_variantes.csv")

            _pct_ok_ref = _dfc["pct_medido_ok"].iloc[0]
            tooltip(f"""
            **¿Por qué casi no cambia el % usable entre la variante total y la selectiva?**

            Ambas usan el mismo modelo ganador; solo difieren en las horas donde YA
            había una medición válida (~{_pct_ok_ref:.1f}% del período completo): la
            variante *total* las reemplaza igual por la predicción del modelo, la
            *selectiva* las conserva tal cual. Como esas horas ya eran válidas de por
            sí, casi no cambian el % de banderas final — la diferencia real entre
            variantes está en la fidelidad a la medición original (la selectiva
            preserva más dato real), no en la tasa de aprobación del QC.
            """)

    # ── Diagnóstico visual — crudo / post-QC / site-adaptado (③) ──────────
    _dg_crudo = _construir_df_etapa_diagnostico(clave, "crudo")
    _dg_postqc = _construir_df_etapa_diagnostico(clave, "post_qc")
    _dg_sa = _construir_df_etapa_diagnostico(clave, "site_adapt")
    if any(d is not None for d in (_dg_crudo, _dg_postqc, _dg_sa)):
        subseccion(f"Diagnóstico visual — {nombre_display}: crudo / post-QC / site-adaptado")
        tooltip(
            "Mismos 3 diagnósticos ya usados en la etapa de control de calidad "
            "(CSI, simetría AM/PM, envolvente), agregando ahora una tercera serie: "
            "③ site-adaptado, el resultado directo de esta etapa (antes del re-QC "
            "final, que es responsabilidad de la etapa de verificación de "
            "desempeño) — así se ve de un vistazo si el site-adaptation efectivamente "
            "mejora estos diagnósticos frente a crudo y post-QC."
        )

        _fig_csi3 = fig_csi_comparativo3_plotly(_dg_crudo, _dg_postqc, _dg_sa, nombre_display)
        if _fig_csi3 is not None:
            st.plotly_chart(_fig_csi3, use_container_width=True, key=f"csi_cmp3_{clave}")
            panel_info(
                "CSI diario mediano en las 3 etapas — si el site-adaptation funciona bien, "
                "la curva verde (③) debería verse más centrada alrededor de 1 que las otras 2, "
                "sin los picos extremos que sí tienen crudo y post-QC."
            )

        _fig_sim3 = fig_simetria_comparativo3_plotly(_dg_crudo, _dg_postqc, _dg_sa, nombre_display)
        if _fig_sim3 is not None:
            st.plotly_chart(_fig_sim3, use_container_width=True, key=f"simetria_cmp3_{clave}")
            panel_info(
                "Asimetría AM/PM en las 3 etapas — una asimetría sistemática que se mantiene "
                "igual en las 3 series es evidencia de un forzante climático real, no un "
                "artefacto que el QC o el site-adaptation deberían corregir."
            )

        subseccion(f"Patrón anual día × hora — {nombre_display}: crudo / post-QC / site-adaptado")
        _fig_hm3 = fig_heatmap_comparativo_plotly(
            (_dg_crudo, _dg_postqc, _dg_sa), ("① Crudo", "② Post-QC", "③ Site-adaptado"),
            "GHI", nombre_display)
        if _fig_hm3 is not None:
            st.plotly_chart(_fig_hm3, use_container_width=True, key=f"heatmap_cmp3_{clave}")
            panel_info(
                "Mapa de calor día del año × hora local, con las 3 etapas compartiendo la MISMA "
                "escala de color. ③ se ve con celdas más uniformes que ① y ②: al venir de un "
                "modelo entrenado con Solcast como referencia (que no tiene huecos), no debería "
                "tener las franjas en blanco que sí muestran crudo y post-QC en los instantes "
                "descartados."
            )
        else:
            st.info(f"No hay datos suficientes de {nombre_display} para este panel.")

        subseccion(f"Envolvente solar — {nombre_display}: crudo / post-QC / site-adaptado")
        _fig_env3 = fig_envolvente_comparativo_plotly(
            (_dg_crudo, _dg_postqc, _dg_sa), ("① Crudo", "② Post-QC", "③ Site-adaptado"),
            "GHI", nombre_display)
        if _fig_env3 is not None:
            st.plotly_chart(_fig_env3, use_container_width=True, key=f"envolvente_cmp3_{clave}")
            panel_info(
                "Las 3 envolventes comparten la MISMA escala de color, para que sean "
                "directamente comparables entre sí. Un \"valle oscuro\" que persiste igual "
                "en las 3 versiones es evidencia de obstrucción real del horizonte (no un "
                "artefacto de datos sucios que el QC/site-adaptation ya deberían corregir). "
                "Las 3 etapas están a la misma resolución nativa de 5 min (todo el proyecto "
                "trabaja a esa resolución, no a promedios de 1h), así que deberían verse "
                "igual de densas entre sí."
            )
        else:
            st.info(f"No hay datos suficientes de {nombre_display} para la envolvente comparativa.")


def render_oe2_resumen_mejoras():
    """Resumen de mejoras por etapa, con una subpestaña por caso de estudio (año).

    Muestra métricas crudo / post-QC / site-adaptado frente a la fuente
    satelital en la misma ventana de prueba, junto con las gráficas de mejora.
    Cada caso tiene una sola estación terrestre, por eso el contenido se
    organiza por año y no compara estaciones lado a lado."""
    seccion("Resumen — mejoras por etapa")
    tooltip(
        "Compara 3 etapas frente a la misma verdad de terreno (el GHI real "
        "medido en la estación), todas en la misma ventana de fechas del "
        "conjunto de prueba:\n\n"
        "- **① Crudo** — el dato tal cual llega, sin ningún filtro.\n"
        "- **② Post-QC** — solo las horas que pasan el control de calidad "
        "propio de la estación.\n"
        "- **③ Site-adaptado** — la predicción del modelo ganador "
        "(entrenado con Solcast + otras variables).\n\n"
        "Si el error no baja en ese orden (①→②→③), algo anda mal en alguna "
        "etapa anterior."
    )

    tablas = cargar_reconstruccion_tablas()
    df_resumen, df_metricas = tablas["resumen"], tablas["metricas"]
    estaciones = [
        ("udenar", "Udenar (terrestre)", C_RED, "2023"),
        (CLAVE_UDENAR_EST, NOMBRE_UDENAR_EST, C_UDENAR_EST, "2025"),
    ]

    _sel_resumen_oe2 = tabs_perezosas([f"📅 {anio}" for *_, anio in estaciones], key="tabs_resumen_oe2_sel")
    for clave, nombre, color, anio in estaciones:
        if _sel_resumen_oe2 == f"📅 {anio}":
            _render_oe2_resumen_una_estacion(clave, nombre, color, df_resumen, df_metricas)


def _render_oe2_resumen_una_estacion(clave, nombre, color, df_resumen, df_metricas):
    """Cuerpo del resumen de mejoras por etapa para una sola estación.

    Se invoca una vez por subpestaña de año desde render_oe2_resumen_mejoras."""
    filas = []
    if df_resumen is not None and clave in df_resumen["estacion"].values:
        fila = df_resumen[df_resumen["estacion"] == clave].iloc[0]
        rango = _parsear_rango_test(fila.get("rango_test", ""))
        if rango is not None:
            ref = calcular_metricas_referencia_ventana(clave, rango[0], rango[1])
            if ref is not None:
                for etq, label in [("crudo", "① Crudo"), ("post_qc", "② Post-QC")]:
                    m = ref.get(etq)
                    if m is None or pd.isna(m.get("MAE")):
                        continue
                    filas.append({"Estación": nombre, "Etapa": label,
                                  "RMSE [W/m²]": round(m["RMSE"], 2),
                                  "MAE [W/m²]": round(m["MAE"], 2),
                                  "R²": round(m["R2"], 4)})
    if df_metricas is not None:
        _dfm_g = df_metricas[(df_metricas["estacion"] == clave) & (df_metricas["modelo_ganador"])]
        if not _dfm_g.empty:
            w = _dfm_g.iloc[0]
            filas.append({"Estación": nombre, "Etapa": f"③ Site-adaptado ({w['modelo']})",
                          "RMSE [W/m²]": round(w["RMSE"], 2),
                          "MAE [W/m²]": round(w["MAE"], 2),
                          "R²": round(w["R2"], 4)})

    if not filas:
        st.warning(f"No hay datos suficientes para el resumen por etapa de {nombre}.")
        return

    df_resumen_show = pd.DataFrame(filas)
    subseccion(f"Tabla — crudo vs. post-QC vs. site-adaptado ({nombre})")
    tabla_html(df_resumen_show)
    boton_descarga_csv(df_resumen_show, f"oe2_resumen_mejoras_por_etapa_{clave}.csv")

    subseccion(f"RMSE por etapa — {nombre}")
    _fig_bar = go.Figure()
    _colores_etapa = {"① Crudo": "#b9c4c9", "② Post-QC": C_GOLD, "③ Site-adaptado": C_AZUL_VIVO}
    _df_bar = df_resumen_show.copy()
    _df_bar["EtapaGrupo"] = _df_bar["Etapa"].where(
        ~_df_bar["Etapa"].str.startswith("③"), "③ Site-adaptado")
    for etapa_grupo in ["① Crudo", "② Post-QC", "③ Site-adaptado"]:
        _sub = _df_bar[_df_bar["EtapaGrupo"] == etapa_grupo]
        if _sub.empty:
            continue
        _fig_bar.add_trace(go.Bar(
            name=etapa_grupo, x=_sub["Estación"], y=_sub["RMSE [W/m²]"],
            marker_color=_colores_etapa[etapa_grupo],
            text=[f"{v:.1f}" for v in _sub["RMSE [W/m²]"]], textposition="outside",
            customdata=_sub["Etapa"],
            hovertemplate="%{x}<br>%{customdata}: %{y:.1f} W/m²<extra></extra>",
        ))
    _fig_bar.update_layout(
        barmode="group", height=380, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=40, b=40, l=40, r=20),
        yaxis=dict(title="RMSE [W/m²]", gridcolor="#e5e5e5"),
    )
    st.plotly_chart(_fig_bar, use_container_width=True, key=f"oe2_resumen_rmse_etapas_{clave}")
    panel_info(
        "En orden (crudo → post-QC → site-adaptado). Lo esperable es una "
        "escalera descendente: el control de calidad ya reduce algo el error "
        "frente a Solcast (descarta datos evidentemente malos), pero la "
        "reducción grande viene del site-adaptation, que aprende y corrige "
        "el sesgo sistemático en vez de solo descartar datos."
    )

    subseccion(f"Dispersión 1:1 — GHI predicho vs. GHI medido, antes y después del site-adaptation ({nombre})")
    tooltip(
        "**¿Qué es lo que se corrige aquí — Solcast o la estación?** Lo que "
        "se corrige/ajusta es **Solcast**, no la estación. La estación es la "
        "\"verdad de terreno\" (el eje X, siempre igual en las dos series); "
        "nunca se le aplica ninguna corrección, porque es el objetivo que "
        "queremos igualar, no algo que haya que arreglar.\n\n"
        "- **Antes** (rojo) = tomar el dato de Solcast **tal cual llega, sin "
        "ningún ajuste**, y usarlo directamente como si fuera el GHI de la "
        "estación. Es el sustituto ingenuo: no se le aplicó ninguna "
        "corrección todavía.\n"
        "- **Después** (azul) = el modelo de site-adaptation toma ese mismo "
        "Solcast (más otras variables) y **lo ajusta/corrige** para que se "
        "parezca mucho más a lo que la estación realmente midió en ese "
        "instante.\n\n"
        "En otras palabras: cuando el sensor de la estación falla o tiene un "
        "hueco, en vez de dejarlo vacío (o de pegar el dato de Solcast sin "
        "más), usamos la versión de Solcast **ya corregida por el modelo** "
        "para rellenar ese hueco — por eso a esta versión se le llama "
        "\"adaptada\" o \"corregida\", y a la otra \"sin corregir\".\n\n"
        "Cuanto más pegados estén los puntos a la diagonal punteada (donde "
        "predicho = medido), mejor. La comparación 'antes vs. después' es la "
        "misma nube de puntos en el eje X (siempre la estación), solo que la "
        "Y es una versión distinta de Solcast en cada caso."
    )
    _df_pred = cargar_procesado(f"reconstruccion_{clave}_predicciones_test.csv")
    col_real = f"GHI_{clave}"
    if (_df_pred is None or df_metricas is None or col_real not in _df_pred.columns
            or "Baseline (Solcast sin corregir)" not in _df_pred.columns):
        st.info(f"No hay datos de predicciones disponibles todavía para {nombre}. "
                "Corre verificar_pieza6c_modelos_4estaciones.py y "
                "verificar_pieza7b_reconstruccion_4estaciones.py.")
        return
    _dfm_g = df_metricas[(df_metricas["estacion"] == clave) & (df_metricas["modelo_ganador"])]
    if _dfm_g.empty:
        st.info(f"No hay modelo ganador registrado todavía para {nombre}.")
        return
    _ganador_col = _dfm_g.iloc[0]["modelo"]
    if _ganador_col not in _df_pred.columns:
        st.info(f"La columna del modelo ganador ({_ganador_col}) no está en las predicciones de {nombre}.")
        return
    _dfp = _df_pred.dropna(subset=[col_real, "Baseline (Solcast sin corregir)", _ganador_col])
    if _dfp.empty:
        st.info(f"No hay filas válidas para graficar la dispersión de {nombre}.")
        return

    # Una sola estación por subpestaña: el gráfico ocupa todo el ancho
    # disponible.
    _fig_disp = go.Figure()
    _fig_disp.add_trace(go.Scatter(
        x=_dfp[col_real], y=_dfp["Baseline (Solcast sin corregir)"],
        mode="markers", name="Antes — Solcast sin corregir",
        marker=dict(color=C_RED, size=5, opacity=0.45)))
    _fig_disp.add_trace(go.Scatter(
        x=_dfp[col_real], y=_dfp[_ganador_col],
        mode="markers", name=f"Después — modelo site-adaptado ({_ganador_col})",
        marker=dict(color=C_AZUL_VIVO, size=5, opacity=0.45)))
    _max_ax = float(np.nanmax([
        _dfp[col_real].max(), _dfp["Baseline (Solcast sin corregir)"].max(), _dfp[_ganador_col].max(),
    ])) * 1.05
    _fig_disp.add_trace(go.Scatter(
        x=[0, _max_ax], y=[0, _max_ax], mode="lines", name="Referencia 1:1",
        line=dict(color="#999", width=1, dash="dash")))

    # Recta de ajuste y pendiente de cada nube de puntos (antes/después), el
    # mismo recurso visual de la literatura de site-adaptation: el ideal es
    # 45° (pendiente = 1). Como el gráfico usa ejes 1:1 (scaleanchor), el
    # ángulo es directamente comparable con el de esas referencias.
    _m_antes, _b_antes = np.polyfit(_dfp[col_real], _dfp["Baseline (Solcast sin corregir)"], 1)
    _m_desp, _b_desp = np.polyfit(_dfp[col_real], _dfp[_ganador_col], 1)
    _ang_antes = np.degrees(np.arctan(_m_antes))
    _ang_desp = np.degrees(np.arctan(_m_desp))
    _x_fit = np.array([0.0, _max_ax])
    _fig_disp.add_trace(go.Scatter(
        x=_x_fit, y=_m_antes * _x_fit + _b_antes, mode="lines",
        name=f"Ajuste antes (pendiente={_m_antes:.2f}, {_ang_antes:.0f}°)",
        line=dict(color=C_RED, width=2, dash="dot")))
    _fig_disp.add_trace(go.Scatter(
        x=_x_fit, y=_m_desp * _x_fit + _b_desp, mode="lines",
        name=f"Ajuste después (pendiente={_m_desp:.2f}, {_ang_desp:.0f}°)",
        line=dict(color=C_AZUL_VIVO, width=2, dash="dot")))

    _fig_disp.update_layout(
        height=560, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        showlegend=True,
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11)),
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis=dict(title="GHI medido en la estación, post-QC [W/m²]", gridcolor="#eee"),
        yaxis=dict(title="GHI predicho [W/m²]", gridcolor="#eee", scaleanchor="x", scaleratio=1),
    )
    st.plotly_chart(_fig_disp, use_container_width=True, key=f"oe2_resumen_dispersion_{clave}",
                     config=PLOTLY_CONFIG)
    panel_info(
        "Cada punto es un instante del conjunto de prueba, con el mismo eje X "
        "(GHI medido) en ambos casos. Mientras más pegados a la diagonal "
        "punteada, mejor. Los puntos rojos (Solcast sin corregir) suelen "
        "desviarse sistemáticamente de la diagonal; los puntos azules "
        "(salida del modelo site-adaptado) deberían agruparse mucho más cerca "
        "de ella. Las rectas punteadas son el ajuste lineal de cada nube y su "
        f"pendiente (45°/pendiente=1 sería el caso ideal): antes de la "
        f"corrección la pendiente es {_m_antes:.2f} ({_ang_antes:.0f}°), y "
        f"después sube a {_m_desp:.2f} ({_ang_desp:.0f}°), acercándose al caso "
        "ideal — esa es la mejora que aporta el site-adaptation, distinta de "
        "lo que ya se revisó en el control de calidad."
    )

    # === ECDF: medido vs. crudo vs. site-adaptado ==========================
    # Gráfica habitual en la literatura de site-adaptation: compara la función
    # de distribución acumulada empírica de las 3 series. Si el site-adaptation
    # funciona, la curva "adaptado" queda prácticamente encima de la de
    # "medido", mientras que la de "crudo" se desvía.
    subseccion(f"ECDF — medido vs. Solcast vs. site-adaptado ({nombre})")
    tooltip(
        "**Cómo leer una ECDF (función de distribución acumulada empírica):** "
        "cada curva responde \"¿qué % de los datos de esta serie son ≤ a este "
        "valor de GHI?\". Por ejemplo, si la curva negra (medido) pasa por "
        "(800, 0.9), quiere decir que el 90% de las mediciones reales fueron "
        "≤ 800 W/m².\n\n"
        "Comparamos 3 curvas: **medido** (la estación, negro — la verdad de "
        "terreno, nunca se le corrige nada), **Solcast** sin corregir (rojo "
        "— Solcast tal cual, sin ningún ajuste) y **Solcast ya corregido por "
        "el modelo** de site-adaptation (azul). Lo que se corrige/ajusta es "
        "siempre Solcast, no la estación. Si el modelo funciona bien, la "
        "curva azul debe quedar prácticamente pegada a la negra en todo el "
        "rango; la curva roja, en cambio, se separa de la negra — esa "
        "separación *es* el error de distribución que corrige el "
        "site-adaptation (lo mismo que mide el indicador KSI).\n\n"
        "El eje Y se recorta desde 0.4 (no desde 0) porque la parte baja de "
        "la curva es sobre todo horas de irradiancia muy baja o nula (noche, "
        "amanecer/atardecer), donde las 3 curvas casi siempre coinciden y no "
        "aportan nada — así se ve mejor la zona donde sí hay diferencias "
        "reales entre las series."
    )

    def _ecdf(valores):
        """Devuelve (x, y) de la función de distribución acumulada empírica."""
        x = np.sort(np.asarray(valores, dtype=float))
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    _x_real, _y_real = _ecdf(_dfp[col_real])
    _x_crudo, _y_crudo = _ecdf(_dfp["Baseline (Solcast sin corregir)"])
    _x_adapt, _y_adapt = _ecdf(_dfp[_ganador_col])
    # Recorte del eje X al percentil 99.5 en lugar del máximo absoluto: unos
    # pocos valores extremos estirarían el eje mucho más allá de la zona donde
    # hay diferencias visibles entre curvas.
    _x_max_ecdf = float(np.nanpercentile(
        np.concatenate([_x_real, _x_crudo, _x_adapt]), 99.5))
    _fig_ecdf = go.Figure()
    _fig_ecdf.add_trace(go.Scatter(x=_x_real, y=_y_real, mode="lines", name="Medido (post-QC)",
        line=dict(color="#1a1a1a", width=2.5)))
    _fig_ecdf.add_trace(go.Scatter(x=_x_crudo, y=_y_crudo, mode="lines",
        name="Solcast", line=dict(color=C_RED, width=1.8, dash="dash")))
    _fig_ecdf.add_trace(go.Scatter(x=_x_adapt, y=_y_adapt, mode="lines",
        name=f"Adaptado ({_ganador_col})", line=dict(color=C_AZUL_VIVO, width=2)))
    _fig_ecdf.update_layout(
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=40, b=40, l=50, r=20),
        xaxis=dict(title="GHI [W/m²]", gridcolor="#eee", range=[0, _x_max_ecdf]),
        yaxis=dict(title="ECDF (-)", gridcolor="#eee", range=[0.4, 1.0]),
    )
    st.plotly_chart(_fig_ecdf, use_container_width=True, key=f"oe2_resumen_ecdf_{clave}")
    panel_info(
        "Función de distribución acumulada empírica de cada serie, sobre todo "
        "el conjunto de prueba. \"Medido (post-QC)\" es el dato de la estación "
        "terrestre que ya pasó el control de calidad (no el crudo) — es la "
        "verdad de terreno contra la que se entrena y evalúa el modelo. Si el "
        "site-adaptation funciona bien, la curva azul (adaptado) debería "
        "quedar prácticamente encima de la negra (medido); la brecha entre la "
        "roja (Solcast, sin corregir) y la negra es la distorsión de "
        "distribución que el modelo corrige."
    )

    # === Serie temporal superpuesta (ventana corta) ========================
    # Superpone medido / crudo / adaptado en una ventana corta para verificar
    # visualmente si el modelo sigue los picos y valles reales mejor que el
    # dato sin corregir.
    subseccion(f"Serie temporal superpuesta — medido vs. crudo vs. adaptado ({nombre})")
    _dfp_ts = _dfp.sort_values("datetime") if "datetime" in _dfp.columns else None
    if _dfp_ts is not None and len(_dfp_ts) > 0:
        _fig_ts = go.Figure()
        _fig_ts.add_trace(go.Scattergl(
            x=_dfp_ts["datetime"], y=_dfp_ts[col_real], mode="lines", name="Medido (post-QC)",
            line=dict(color="#1a1a1a", width=1.6)))
        _fig_ts.add_trace(go.Scattergl(
            x=_dfp_ts["datetime"], y=_dfp_ts["Baseline (Solcast sin corregir)"], mode="lines",
            name="Solcast", line=dict(color=C_RED, width=1.2, dash="dash")))
        _fig_ts.add_trace(go.Scattergl(
            x=_dfp_ts["datetime"], y=_dfp_ts[_ganador_col], mode="lines",
            name=f"Adaptado ({_ganador_col})", line=dict(color=C_AZUL_VIVO, width=1.4)))
        # Vista inicial de ~10 días: el conjunto de prueba completo puede
        # abarcar meses y a resolución de 5 min se vería como una masa sólida.
        # El selector de rango / control deslizante permite acercarse o ver todo.
        _t0 = _dfp_ts["datetime"].min()
        _t_fin_vista = min(_t0 + pd.Timedelta(days=10), _dfp_ts["datetime"].max())
        _fig_ts.update_layout(
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial", size=11),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(t=40, b=20, l=50, r=20),
            xaxis=dict(
                gridcolor="#eee", range=[_t0, _t_fin_vista],
                rangeselector=dict(buttons=[
                    dict(count=10, label="10D", step="day", stepmode="backward"),
                    dict(count=30, label="30D", step="day", stepmode="backward"),
                    dict(step="all", label="Todo"),
                ], bgcolor="#f0f6f9", activecolor="#1F4E5F", font=dict(size=10)),
                rangeslider=dict(visible=True, thickness=0.07),
            ),
            yaxis=dict(title="GHI [W/m²]", gridcolor="#eee", rangemode="tozero"),
        )
        st.plotly_chart(_fig_ts, use_container_width=True, key=f"oe2_resumen_serie_superpuesta_{clave}")
        panel_info(
            "Ventana inicial de ~10 días del conjunto de prueba (usa el "
            "selector de rango o el control deslizante para ver otros "
            "periodos, o 'Todo' para el conjunto completo). \"Medido "
            "(post-QC)\" es el dato de la estación que ya pasó el control de "
            "calidad. Compara visualmente si el modelo (línea azul) sigue de "
            "cerca los picos y valles reales (línea negra) mejor que Solcast "
            "sin corregir (línea roja)."
        )
    else:
        st.info(f"No hay columna de fecha/hora disponible para graficar la serie superpuesta de {nombre}.")


def fig_diagrama_baseline() -> go.Figure:
    """Ilustra por qué el baseline (GHI_pred = GHI satelital) tiene error:
    compara la línea 1:1 (sin corrección) contra una relación real simulada
    con sesgo. Datos sintéticos, solo para ilustrar el concepto."""
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(50, 950, 90))
    y = 0.72 * x + 15 + rng.normal(0, 35, len(x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Relación real (simulada)",
        marker=dict(color="#b9c4c9", size=6, opacity=0.7)))
    xr = np.array([0, 1000])
    fig.add_trace(go.Scatter(x=xr, y=xr, mode="lines", name="Predicción del baseline (y = x)",
        line=dict(color=C_RED, width=2, dash="dash")))
    b1, b0 = np.polyfit(x, y, 1)
    fig.add_trace(go.Scatter(x=xr, y=b1 * xr + b0, mode="lines", name="Tendencia real",
        line=dict(color=C_AZUL_VIVO, width=2.5)))
    x0 = 700
    fig.add_annotation(x=x0, y=b1 * x0 + b0, ax=x0, ay=x0, axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowcolor=C_GOLD, arrowwidth=2, text="")
    fig.add_annotation(x=x0 + 15, y=(x0 + b1 * x0 + b0) / 2, text="error del<br>baseline",
        showarrow=False, font=dict(size=10, color=C_GOLD), xanchor="left")
    fig.update_layout(height=330, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        legend=dict(orientation="h", y=1.15, x=0),
        margin=dict(t=30, b=40, l=40, r=20),
        xaxis=dict(title="GHI Solcast [W/m²]", gridcolor="#e5e5e5"),
        yaxis=dict(title="GHI estación [W/m²]", gridcolor="#e5e5e5"))
    return fig


def fig_diagrama_regresion_simple() -> go.Figure:
    """Ilustra la regresión lineal simple: recta ajustada con beta0 (intercepto)
    y beta1 (pendiente) marcados sobre datos sintéticos ilustrativos."""
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(50, 950, 90))
    y = 0.72 * x + 15 + rng.normal(0, 35, len(x))
    b1, b0 = np.polyfit(x, y, 1)
    xr = np.array([0, 1000])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Datos de entrenamiento",
        marker=dict(color="#b9c4c9", size=6, opacity=0.7)))
    fig.add_trace(go.Scatter(x=xr, y=b1 * xr + b0, mode="lines",
        name=f"Recta ajustada: y = {b0:.0f} + {b1:.2f}x",
        line=dict(color=C_AZUL_VIVO, width=2.5)))
    fig.add_trace(go.Scatter(x=[0], y=[b0], mode="markers+text", showlegend=False,
        marker=dict(color=C_GOLD, size=10), text=["β₀"], textposition="middle left"))
    xa, xb = 300, 600
    ya, yb = b1 * xa + b0, b1 * xb + b0
    fig.add_shape(type="line", x0=xa, y0=ya, x1=xb, y1=ya, line=dict(color=C_GOLD, width=1, dash="dot"))
    fig.add_shape(type="line", x0=xb, y0=ya, x1=xb, y1=yb, line=dict(color=C_GOLD, width=1, dash="dot"))
    fig.add_annotation(x=(xa + xb) / 2, y=ya - 25, text="Δx", showarrow=False, font=dict(size=10, color=C_GOLD))
    fig.add_annotation(x=xb + 25, y=(ya + yb) / 2, text="Δy → β₁ = Δy/Δx", showarrow=False,
        font=dict(size=10, color=C_GOLD), xanchor="left")
    fig.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        legend=dict(orientation="h", y=1.15, x=0),
        margin=dict(t=30, b=40, l=40, r=20),
        xaxis=dict(title="GHI Solcast [W/m²]", gridcolor="#e5e5e5"),
        yaxis=dict(title="GHI estación [W/m²]", gridcolor="#e5e5e5"))
    return fig


def fig_diagrama_regresion_multivariable() -> go.Figure:
    """Diagrama de nodos: varias variables predictoras entran con un peso
    (beta_i) cada una a una suma ponderada que produce GHI_pred."""
    variables = ["GHI solcast", "DNI", "DHI", "Temp.", "Humedad", "Viento",
                 "Áng. cenital", "Hora del día"]
    ys = np.linspace(0.92, 0.08, len(variables))
    fig = go.Figure()
    for i, (var, y) in enumerate(zip(variables, ys)):
        fig.add_shape(type="rect", x0=0.02, x1=0.32, y0=y - 0.045, y1=y + 0.045,
            line=dict(color=C_AZUL_VIVO, width=1.5), fillcolor=C_LIGHT)
        fig.add_annotation(x=0.17, y=y, text=var, showarrow=False, font=dict(size=10, color=C_AZUL_VIVO))
        fig.add_annotation(x=0.66, y=0.5, ax=0.32, ay=y, axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor="#9fb0b8", text="")
        fig.add_annotation(x=(0.32 + 0.66) / 2, y=(y + 0.5) / 2 + 0.015, text=f"β{i+1}",
            showarrow=False, font=dict(size=9, color=C_GOLD))
    fig.add_shape(type="circle", x0=0.66, x1=0.80, y0=0.42, y1=0.58,
        line=dict(color=C_RED, width=2), fillcolor="white")
    fig.add_annotation(x=0.73, y=0.5, text="Σ", showarrow=False, font=dict(size=20, color=C_RED))
    fig.add_annotation(x=0.73, y=0.74, ax=0.73, ay=0.58, axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor=C_GOLD, text="")
    fig.add_annotation(x=0.73, y=0.80, text="β₀ (sesgo)", showarrow=False, font=dict(size=9, color=C_GOLD))
    fig.add_annotation(x=0.95, y=0.5, ax=0.80, ay=0.5, axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=C_RED, text="")
    fig.add_annotation(x=0.97, y=0.5, text="GHI<sub>pred</sub>", showarrow=False,
        font=dict(size=12, color=C_RED), xanchor="left")
    fig.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11), margin=dict(t=20, b=20, l=10, r=70),
        xaxis=dict(visible=False, range=[0, 1.1]), yaxis=dict(visible=False, range=[0, 1]))
    return fig


def fig_diagrama_eqm() -> go.Figure:
    """Ilustra el mapeo por percentiles (EQM): dos distribuciones empíricas
    (fuente satelital y estación, simuladas) y cómo un mismo percentil se traduce en
    valores distintos de GHI en cada una."""
    rng = np.random.default_rng(11)
    sol = np.sort(rng.gamma(4, 90, 400))
    est = np.sort(rng.gamma(3, 75, 400))
    p_sol = np.linspace(0, 100, len(sol))
    p_est = np.linspace(0, 100, len(est))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sol, y=p_sol, mode="lines", name="CDF Solcast (entrenamiento)",
        line=dict(color=C_AZUL_VIVO, width=2.5)))
    fig.add_trace(go.Scatter(x=est, y=p_est, mode="lines", name="CDF estación (entrenamiento)",
        line=dict(color=C_RED, width=2.5)))
    p_ejemplo = 70
    x_sol_p = np.interp(p_ejemplo, p_sol, sol)
    x_est_p = np.interp(p_ejemplo, p_est, est)
    fig.add_shape(type="line", x0=0, x1=max(x_sol_p, x_est_p) + 40, y0=p_ejemplo, y1=p_ejemplo,
        line=dict(color=C_GOLD, width=1, dash="dot"))
    fig.add_shape(type="line", x0=x_sol_p, x1=x_sol_p, y0=0, y1=p_ejemplo,
        line=dict(color=C_AZUL_VIVO, width=1, dash="dot"))
    fig.add_shape(type="line", x0=x_est_p, x1=x_est_p, y0=0, y1=p_ejemplo,
        line=dict(color=C_RED, width=1, dash="dot"))
    fig.add_annotation(x=x_sol_p, y=-6, text=f"{x_sol_p:.0f}", showarrow=False,
        font=dict(size=10, color=C_AZUL_VIVO))
    fig.add_annotation(x=x_est_p, y=-6, text=f"{x_est_p:.0f}", showarrow=False,
        font=dict(size=10, color=C_RED))
    fig.add_annotation(x=x_est_p, y=p_ejemplo, ax=x_sol_p, ay=p_ejemplo, axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowcolor=C_GOLD, arrowwidth=2, text="")
    fig.add_annotation(x=(x_sol_p + x_est_p) / 2, y=p_ejemplo + 5,
        text=f"percentil {p_ejemplo} en ambas → {x_sol_p:.0f} se mapea a {x_est_p:.0f}",
        showarrow=False, font=dict(size=10, color=C_GOLD))
    fig.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        legend=dict(orientation="h", y=1.15, x=0),
        margin=dict(t=30, b=40, l=40, r=20),
        xaxis=dict(title="GHI [W/m²]", gridcolor="#e5e5e5"),
        yaxis=dict(title="Percentil [%]", gridcolor="#e5e5e5", range=[-10, 105]))
    return fig


def fig_diagrama_arbol_individual() -> go.Figure:
    """Diagrama de UN árbol de decisión ilustrativo (parámetros de ejemplo, no
    ajustados a datos reales): Random Forest promedia 300 árboles como este,
    cada uno entrenado con una muestra bootstrap distinta."""
    fig = go.Figure()
    nodos = [
        dict(x=0.5, y=0.90, texto="GHI solcast<br>< 350?", tipo="decision"),
        dict(x=0.25, y=0.58, texto="hora<br>< 10?", tipo="decision"),
        dict(x=0.75, y=0.58, texto="áng. cenital<br>< 35°?", tipo="decision"),
        dict(x=0.10, y=0.22, texto="GHI≈90<br>W/m²", tipo="hoja"),
        dict(x=0.40, y=0.22, texto="GHI≈210<br>W/m²", tipo="hoja"),
        dict(x=0.60, y=0.22, texto="GHI≈480<br>W/m²", tipo="hoja"),
        dict(x=0.90, y=0.22, texto="GHI≈620<br>W/m²", tipo="hoja"),
    ]
    conexiones = [(0, 1, "Sí"), (0, 2, "No"), (1, 3, "Sí"), (1, 4, "No"),
                  (2, 5, "Sí"), (2, 6, "No")]
    for i, j, etiqueta in conexiones:
        a, b = nodos[i], nodos[j]
        fig.add_shape(type="line", x0=a["x"], y0=a["y"] - 0.06, x1=b["x"], y1=b["y"] + 0.06,
            line=dict(color="#9fb0b8", width=1.3))
        fig.add_annotation(x=(a["x"] + b["x"]) / 2, y=(a["y"] + b["y"]) / 2, text=etiqueta,
            showarrow=False, font=dict(size=9, color=C_GOLD), bgcolor="white")
    for nodo in nodos:
        color = C_AZUL_VIVO if nodo["tipo"] == "decision" else C_RED
        fig.add_shape(type="rect", x0=nodo["x"] - 0.11, x1=nodo["x"] + 0.11,
            y0=nodo["y"] - 0.06, y1=nodo["y"] + 0.06,
            line=dict(color=color, width=1.8),
            fillcolor=C_LIGHT if nodo["tipo"] == "decision" else "#fbe4e6")
        fig.add_annotation(x=nodo["x"], y=nodo["y"], text=nodo["texto"], showarrow=False,
            font=dict(size=9.5, color=color))
    fig.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11), margin=dict(t=20, b=10, l=10, r=10),
        xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0.1, 1]))
    return fig


def fig_diagrama_extratrees_corte() -> go.Figure:
    """Ilustra la diferencia clave entre Random Forest y Extra Trees: cómo se
    elige el punto de corte dentro de un nodo del árbol."""
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(100, 900, 40))
    y = np.zeros_like(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Datos en este nodo",
        marker=dict(color="#b9c4c9", size=8, opacity=0.8)))
    for c in (320, 480, 620):
        fig.add_shape(type="line", x0=c, x1=c, y0=-0.5, y1=0.5,
            line=dict(color="#9fb0b8", width=1, dash="dot"))
    fig.add_shape(type="line", x0=480, x1=480, y0=-0.5, y1=0.5, line=dict(color=C_AZUL_VIVO, width=3))
    fig.add_annotation(x=480, y=0.7,
        text="Random Forest:<br>evalúa TODOS los cortes<br>y elige el de menor error",
        showarrow=False, font=dict(size=9.5, color=C_AZUL_VIVO))
    fig.add_shape(type="line", x0=320, x1=320, y0=-0.5, y1=0.5, line=dict(color=C_RED, width=3))
    fig.add_annotation(x=320, y=-0.75,
        text="Extra Trees:<br>sortea el corte entre<br>candidatos aleatorios",
        showarrow=False, font=dict(size=9.5, color=C_RED))
    fig.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11), margin=dict(t=40, b=50, l=30, r=30),
        showlegend=False,
        xaxis=dict(title="GHI solcast [W/m²] (variable del corte)", gridcolor="#e5e5e5"),
        yaxis=dict(visible=False, range=[-1.1, 1.1]))
    return fig


def fig_diagrama_gradient_boosting() -> go.Figure:
    """Ilustra el boosting secuencial: cada árbol nuevo corrige el residual
    que dejó la suma de los árboles anteriores."""
    fig = go.Figure()
    etapas = [
        dict(x=0.15, texto="Árbol 1<br>predice GHI"),
        dict(x=0.45, texto="Árbol 2<br>ajusta el<br>residual de 1"),
        dict(x=0.75, texto="Árbol 3<br>ajusta el<br>residual de 1+2"),
    ]
    for i, e in enumerate(etapas):
        fig.add_shape(type="rect", x0=e["x"] - 0.11, x1=e["x"] + 0.11, y0=0.62, y1=0.88,
            line=dict(color=C_AZUL_VIVO, width=1.8), fillcolor=C_LIGHT)
        fig.add_annotation(x=e["x"], y=0.75, text=e["texto"], showarrow=False,
            font=dict(size=9.5, color=C_AZUL_VIVO))
        if i > 0:
            xprev = etapas[i - 1]["x"]
            fig.add_annotation(x=e["x"] - 0.11, y=0.75, ax=xprev + 0.11, ay=0.75,
                axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=1.3,
                arrowcolor=C_GOLD, text="")
            fig.add_annotation(x=(xprev + e["x"]) / 2, y=0.93, text="residual", showarrow=False,
                font=dict(size=9, color=C_GOLD))
    fig.add_shape(type="circle", x0=0.44, x1=0.56, y0=0.18, y1=0.38,
        line=dict(color=C_RED, width=2), fillcolor="white")
    fig.add_annotation(x=0.5, y=0.28, text="Σ", showarrow=False, font=dict(size=20, color=C_RED))
    for e in etapas:
        fig.add_annotation(x=0.5, y=0.38, ax=e["x"], ay=0.62, axref="x", ayref="y",
            showarrow=True, arrowhead=1, arrowwidth=1, arrowcolor="#9fb0b8", text="")
    fig.add_annotation(x=0.5, y=0.04, text="GHI_pred = Σ (predicción de cada árbol)",
        showarrow=False, font=dict(size=11, color=C_RED))
    fig.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11), margin=dict(t=30, b=10, l=10, r=10),
        xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1]))
    return fig


C_VERDE_SALIDA = "#2e7d32"


def fig_diagrama_site_adaptation_general() -> go.Figure:
    """Diagrama general del esquema de site-adaptation en dos filas.

    (1) Entrenamiento: la fuente satelital (X) y la estación terrestre (y, solo
    horas confiables) alimentan los 9 modelos candidatos; el ganador es el
    modelo de site-adaptation de esa estación. (2) Aplicación: ese mismo
    modelo recibe la serie satelital del año completo y produce el dataset
    reconstruido/adaptado."""
    fig = go.Figure()

    # ---- Fila superior: ENTRENAMIENTO ----
    fig.add_shape(type="rect", x0=0.02, x1=0.30, y0=0.80, y1=0.98,
        line=dict(color=C_AZUL_VIVO, width=1.8), fillcolor=C_LIGHT)
    fig.add_annotation(x=0.16, y=0.89, text="Solcast (satelital)<br><b>X</b>: variables predictoras",
        showarrow=False, font=dict(size=10, color=C_AZUL_VIVO))

    fig.add_shape(type="rect", x0=0.02, x1=0.30, y0=0.58, y1=0.76,
        line=dict(color=C_RED, width=1.8), fillcolor=C_LIGHT)
    fig.add_annotation(x=0.16, y=0.67, text="Estación terrestre<br><b>y</b>: GHI confiable (post-QC)",
        showarrow=False, font=dict(size=10, color=C_RED))

    fig.add_shape(type="rect", x0=0.42, x1=0.68, y0=0.64, y1=0.92,
        line=dict(color=C_TEAL, width=2, dash="dash"), fillcolor="white")
    fig.add_annotation(x=0.55, y=0.78, text="<b>Entrenamiento ML</b><br>9 modelos candidatos<br>"
        "(70% train / 30% test)", showarrow=False, font=dict(size=10, color=C_TEAL))

    for y0 in (0.89, 0.67):
        fig.add_annotation(x=0.42, y=0.78, ax=0.30, ay=y0, axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowwidth=1.3, arrowcolor="#9fb0b8", text="")

    fig.add_shape(type="rect", x0=0.80, x1=1.06, y0=0.66, y1=0.90,
        line=dict(color=C_GOLD, width=2.4), fillcolor="#fff8e8")
    fig.add_annotation(x=0.93, y=0.78, text="<b>Modelo de<br>site-adaptation</b><br>(el ganador,<br>por estación)",
        showarrow=False, font=dict(size=10, color="#7a5b00"))
    fig.add_annotation(x=0.80, y=0.78, ax=0.68, ay=0.78, axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=C_TEAL, text="")
    fig.add_annotation(x=0.74, y=0.95, text="filtro físico +<br>menor RMSE", showarrow=False,
        font=dict(size=8.5, color=C_TEAL))

    fig.add_annotation(x=0.02, y=1.0, text="① ENTRENAMIENTO — el resultado es el modelo, no una predicción suelta",
        showarrow=False, font=dict(size=10.5, color=C_TEAL, family="Arial Black"), xanchor="left")

    # ---- Conector vertical: mismo modelo, ya entrenado ----
    fig.add_annotation(x=0.93, y=0.32, ax=0.93, ay=0.66, axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowwidth=1.6, arrowcolor=C_GOLD, arrowside="end",
        text="", standoff=2)
    fig.add_annotation(x=1.00, y=0.49, text="mismo modelo,<br>ya entrenado<br>(no se reentrena)",
        showarrow=False, font=dict(size=8.5, color="#7a5b00"), xanchor="left")

    # ---- Fila inferior: APLICACIÓN ----
    fig.add_shape(type="rect", x0=0.02, x1=0.30, y0=0.14, y1=0.34,
        line=dict(color=C_AZUL_VIVO, width=1.8), fillcolor=C_LIGHT)
    fig.add_annotation(x=0.16, y=0.24, text="Solcast completo<br>(104.825 datos, año entero, 5 min)",
        showarrow=False, font=dict(size=10, color=C_AZUL_VIVO))

    fig.add_shape(type="rect", x0=0.80, x1=1.06, y0=0.12, y1=0.36,
        line=dict(color=C_GOLD, width=2.4), fillcolor="#fff8e8")
    fig.add_annotation(x=0.93, y=0.24, text="<b>Modelo de<br>site-adaptation</b><br>(aplicado)",
        showarrow=False, font=dict(size=10, color="#7a5b00"))
    fig.add_annotation(x=0.80, y=0.24, ax=0.30, ay=0.24, axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowwidth=1.3, arrowcolor="#9fb0b8", text="")

    fig.add_shape(type="rect", x0=1.14, x1=1.42, y0=0.10, y1=0.38,
        line=dict(color=C_VERDE_SALIDA, width=2), fillcolor="#eef7ee")
    fig.add_annotation(x=1.28, y=0.24,
        text="<b>Dataset<br>reconstruido</b><br>solo huecos +<br>horas con flag",
        showarrow=False, font=dict(size=9.5, color=C_VERDE_SALIDA))
    fig.add_annotation(x=1.14, y=0.24, ax=1.06, ay=0.24, axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=C_VERDE_SALIDA, text="")

    fig.add_annotation(x=0.02, y=0.38, text="② APLICACIÓN — el modelo entrenado corrige/rellena, no vuelve a aprender",
        showarrow=False, font=dict(size=10.5, color="#7a5b00", family="Arial Black"), xanchor="left")

    fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11), margin=dict(t=15, b=10, l=10, r=10),
        xaxis=dict(visible=False, range=[0, 1.45]), yaxis=dict(visible=False, range=[0, 1.05]))
    return fig


def render_oe2_marco_metodologico():
    """Marco metodológico de OE2: cómo funcionan los 9 modelos de reconstrucción,
    con un ejemplo real del caso de estudio por modelo. Sigue el mismo estilo
    que el marco metodológico de QC en OE1."""
    seccion("Marco metodológico — Modelos de reconstrucción (site-adaptation)")

    subseccion("¿Qué es, exactamente, el modelo de site-adaptation?")
    st.markdown("""
    El esquema de site-adaptation tiene **dos etapas separadas**, no una sola
    predicción: primero se **entrena y selecciona** un modelo (Solcast +
    estación → modelo ganador), y después ese modelo **ya entrenado** se
    **aplica** sobre Solcast para reconstruir la serie de la estación. El
    diagrama de abajo muestra las dos etapas con los nombres reales de este
    proyecto.
    """)
    st.plotly_chart(fig_diagrama_site_adaptation_general(), use_container_width=True,
                     key="oe2_diagrama_general_site_adapt")
    panel_info("""
    **① Entrenamiento:** Solcast (satelital) es la variable predictora (**X**);
    el GHI de la estación terrestre, solo en las muestras del conjunto
    *supervisado* (donde ya pasó el control de calidad de OE1), es la variable
    objetivo (**y**). Se entrena con el 70% inicial de ese conjunto y se
    evalúa con el 30% final, comparando 9 modelos candidatos. El que gane
    —tras el filtro de validez física y el criterio de menor RMSE— **se
    convierte en EL modelo de site-adaptation de esa estación**. El resultado
    de esta etapa es un modelo entrenado y seleccionado, no una predicción
    suelta.

    **② Aplicación:** ese mismo modelo, ya entrenado (no se reentrena),
    recibe ahora Solcast del año completo (las 104.825 muestras a 5 min,
    incluidas las que la estación nunca midió o midió mal) y genera el
    dataset reconstruido. En la variante *selectiva*, el modelo solo
    reemplaza las muestras marcadas por el QC o sin dato — las que la
    estación ya medía bien no se tocan.

    **Ejemplo real (Udenar 2023):** en el entrenamiento, la regresión lineal
    simple ganó con RMSE = 98.60 W/m² sobre las 15.725 muestras de prueba (a
    resolución nativa de 5 min, evaluado contra el GHI real de Udenar, nunca
    visto en entrenamiento). Esa misma recta, con sus coeficientes ya fijos
    (GHI = 0.494 + 0.735·GHI_Solcast), es la que se aplicó después sobre las
    104.825 muestras de Solcast del año completo (cada 5 min) para producir
    `GHI_corregido_total`.

    **Una aclaración importante, para no confundir dos comparaciones
    distintas:** el RMSE que decide el modelo ganador (arriba) **no es la
    misma métrica** que el %usable de OE1/OE3, ni la comparación
    crudo-vs-adaptado:
    - *Selección del modelo* (esta sección): error de regresión (MAE, RMSE,
      R²) del modelo contra el GHI **real** de la estación, en el conjunto
      de prueba — mide qué tan bien predice.
    - *Control de calidad* (OE1 y OE3): banderas físicas/expertas (nocturno,
      límites físicos, persistencia, etc.) aplicadas a una serie —cruda o
      reconstruida— para decidir si cada dato es físicamente aceptable, sin
      comparar contra ninguna referencia externa.

    Son preguntas distintas — por eso el proyecto las verifica por separado,
    y un modelo con buen RMSE en esta sección igual pasa por el re-QC
    completo de OE3 antes de darlo por bueno.
    """)

    st.markdown("---")
    st.markdown("""
    Los 9 modelos comparten la **misma interfaz** (`fit(X, y)` / `predict(X)`),
    para poder entrenarlos y evaluarlos en igualdad de condiciones:

    - **X (predictoras):** GTI, GHI, DHI, DNI, temperatura, humedad, viento y
      opacidad de nubes de **Solcast**, más geometría solar (ángulo cenital,
      elevación) y variables temporales (hora, día del año, mes).
    - **y (objetivo):** GHI medido de la estación terrestre.
    - **Entrenamiento:** 70% inicial del conjunto *supervisado* (horas con
      medición confiable, sin fuga temporal). **Evaluación:** 30% final.
    - **Aplicación:** el modelo ganador se aplica sobre **todo** el período
      disponible (no solo las horas de entrenamiento/prueba) para reconstruir
      la serie completa — así es como se rellenan los huecos y se corrigen
      las horas flageadas.

    De los 9, se descarta cualquier modelo cuyas predicciones sobre **todo**
    el período de aplicación violen las pruebas físicas (valor bajo el mínimo
    de −4 W/m², límites físicos o irradiancia nocturna, con una tolerancia
    nocturna de 0,01 %). Entre los que pasan el filtro, gana el de menor RMSE
    en prueba.
    """)

    with st.expander("📉 Modelo 1 — Baseline (Solcast sin corregir)", expanded=True):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Cómo funciona:**")
            st.latex(r"GHI_{pred} = GHI_{solcast}")
            st.markdown("""
            No aprende nada: devuelve el dato de Solcast tal cual, sin ningún
            ajuste. Es la **vara de comparación** — el "antes" que cualquier
            modelo entrenado debe superar para justificar su uso.

            **Como candidato a modelo de site-adaptation:** nunca puede serlo
            en la práctica — no corrige nada, por lo que jamás gana el
            criterio de menor RMSE. Su único rol es servir de referencia
            para medir cuánto aportan los demás candidatos.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, prueba):**")
            st.info("""
            MAE = **64.54 W/m²** · RMSE = **123.64 W/m²**

            Comparado con Udenar adaptado (modelo ganador, regresión lineal
            simple: MAE 48.21, RMSE 98.60), el baseline tiene ~25.4% más error (RMSE) —
            confirma que Solcast necesita corrección local antes de usarse
            como estimador del recurso en Udenar.
            """)

        st.markdown("**Cómo se ve (datos simulados para ilustrar el concepto):**")
        st.plotly_chart(fig_diagrama_baseline(), use_container_width=True, key="oe2_diagrama_baseline")
        panel_info(
            "La línea roja punteada es lo que hace el baseline: asume que la "
            "estación mide exactamente lo mismo que Solcast (y = x). La línea "
            "azul es la relación real simulada — la distancia vertical entre "
            "ambas, en cualquier punto, es el error que el baseline no corrige."
        )

    with st.expander("📈 Modelo 2 — Regresión lineal simple"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Cómo funciona (mínimos cuadrados ordinarios):**")
            st.latex(r"GHI_{estación} = \beta_0 + \beta_1 \cdot GHI_{solcast}")
            st.markdown("""
            Ajusta una sola recta entre el GHI de Solcast y el de la estación.
            Es el modelo de site-adaptation más simple e interpretable de la
            literatura clásica. En teoría una recta **puede extrapolar** fuera
            del rango de entrenamiento, por eso su validez física se comprueba
            empíricamente sobre todo el período de aplicación, igual que la de
            los demás modelos.

            **Como candidato a modelo de site-adaptation:** con Solcast como
            única predictora y pendiente < 1, sus predicciones quedan acotadas
            en la práctica (mínimo 0.49 W/m² en Udenar 2023 y 0.14 W/m² en
            Udenar 2025). Pasa el filtro de validez física y, en ambos casos,
            es el modelo ganador.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, prueba):**")
            st.info("""
            RMSE = **98.60 W/m²** — al nivel de los árboles de bagging
            (98.66-98.67 W/m²) y sin ninguna predicción inválida en todo el
            período de aplicación (0 de 104.825 muestras).

            Resumen (V11): RMSE = 98.60 W/m² en Udenar 2023 y
            111.43 W/m² en Udenar 2025, con 0 % de predicciones inválidas en
            todo el período de aplicación. Es el **modelo ganador** en ambos
            conjuntos: la corrección que necesita Solcast en el sitio es
            principalmente de escala (pendiente 0.73 y 0.78).
            """)

        st.markdown("**Cómo se ve (datos simulados para ilustrar el concepto):**")
        st.plotly_chart(fig_diagrama_regresion_simple(), use_container_width=True, key="oe2_diagrama_reg_simple")
        panel_info(
            "β₀ es el punto donde la recta cruza el eje vertical (el valor "
            "predicho cuando Solcast marca 0). β₁ es la pendiente: cuánto sube "
            "la predicción de la estación por cada W/m² que sube Solcast. Con "
            "β₁ < 1 (como en este ejemplo), el modelo 'aplana' los valores "
            "altos de Solcast — típico cuando Solcast tiende a sobreestimar."
        )

    with st.expander("📈 Modelo 3 — Regresión lineal multivariable"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Cómo funciona:**")
            st.latex(r"GHI_{estación} = \beta_0 + \sum_{i=1}^{13} \beta_i \cdot X_i")
            st.markdown("""
            Igual que el modelo 2, pero usa las 13 variables predictoras
            disponibles (no solo GHI de Solcast). Captura más relaciones,
            pero **hereda el mismo riesgo de extrapolación** — con más
            variables, el riesgo de combinaciones fuera de rango es mayor,
            no menor.

            **Como candidato a modelo de site-adaptation:** es el ejemplo más
            claro de por qué el RMSE en prueba nunca es, por sí solo,
            suficiente para elegir el modelo que se va a desplegar — un buen
            ajuste estadístico no garantiza que la salida sea físicamente
            utilizable en las 104.825 muestras (todo el período de
            aplicación, a 5 min) donde después se va a aplicar.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, prueba):**")
            st.error("""
            RMSE = **98.30 W/m²** — competitivo con los árboles de bagging.

            Pero al aplicarlo sobre TODO el período de aplicación (104.825
            muestras), produce GHI **físicamente inválido en 32.431
            instantes (30.9%)** — incluyendo 18.536 (17.6%) por debajo del
            mínimo físico simple. Se excluye por el filtro empírico
            riguroso (validez física sobre el año completo): es el ejemplo
            más claro de por qué la validez física debe comprobarse sobre
            todo el período de aplicación y no solo en la prueba.
            """)

        st.markdown("**Cómo se ve — esquema de la combinación ponderada:**")
        st.plotly_chart(fig_diagrama_regresion_multivariable(), use_container_width=True,
                         key="oe2_diagrama_reg_multivariable")
        panel_info(
            "A diferencia del modelo anterior (una sola recta), aquí cada "
            "variable predictora entra con su propio peso (β₁, β₂, ...) y se "
            "suman junto con un sesgo base (β₀) para producir la predicción "
            "final. Más variables capturan más relaciones, pero cada peso "
            "adicional es otra forma de que la combinación se salga del rango "
            "físicamente razonable en datos nuevos — de ahí el riesgo de "
            "extrapolación mencionado a la izquierda."
        )

    with st.expander("🔀 Modelo 4 — EQM (Empirical Quantile Mapping)"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Cómo funciona:**")
            st.latex(r"GHI_{pred} = F^{-1}_{estación}\big(F_{solcast}(GHI_{solcast})\big)")
            st.markdown("""
            En vez de ajustar una relación funcional punto-a-punto, mapea
            **percentil a percentil**: calcula 100 percentiles de Solcast y
            de la estación en entrenamiento, y para un dato nuevo busca su
            percentil en Solcast e interpola el valor de la estación en ese
            mismo percentil. Corrige sesgo **y forma de la distribución**,
            no solo la media — por eso es robusto cuando la relación
            punto-a-punto es ruidosa.

            ```python
            q_solcast  = np.percentile(GHI_solcast_train, [0..100])
            q_estacion = np.percentile(GHI_estacion_train, [0..100])
            pred = np.interp(GHI_solcast_nuevo, q_solcast, q_estacion)
            ```

            **Como candidato a modelo de site-adaptation:** su fortaleza es
            la robustez frente al ruido (no asume una relación funcional
            exacta punto a punto, solo iguala distribuciones); su debilidad
            es que ignora toda variable adicional (temperatura, geometría
            solar, etc.) — solo usa la distribución marginal de GHI de
            Solcast, nada más.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, prueba):**")
            st.info("""
            RMSE = **104.57 W/m²** — puesto 8 de 9, solo por delante de la
            referencia sin corregir y por detrás de las dos regresiones
            lineales, los modelos de árboles y la red neuronal.

            En Udenar la relación con Solcast es principalmente de escala,
            así que una recta o un modelo de regresión la capturan mejor que
            igualar distribuciones; EQM tiende a destacar más en estaciones
            con relaciones más ruidosas, donde igualar distribuciones es
            más robusto que ajustar la relación punto a punto.
            """)

        st.markdown("**Cómo se ve (distribuciones simuladas para ilustrar el concepto):**")
        st.plotly_chart(fig_diagrama_eqm(), use_container_width=True, key="oe2_diagrama_eqm")
        panel_info(
            "Cada curva es la distribución acumulada (qué % de las horas de "
            "entrenamiento está por debajo de cada valor de GHI) de Solcast "
            "(azul) y de la estación (rojo). Para corregir un dato nuevo de "
            "Solcast, EQM busca en qué percentil cae dentro de la curva azul, "
            "y devuelve el valor de GHI que está en ese mismo percentil de la "
            "curva roja — así ajusta sesgo y forma de la distribución a la vez, "
            "sin asumir que la relación es una línea recta."
        )

    with st.expander("🌲 Modelo 5 — Random Forest"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Cómo funciona (bagging de árboles):**")
            st.markdown("""
            Entrena **300 árboles de decisión** independientes, cada uno
            sobre una muestra bootstrap de los datos de entrenamiento, y
            promedia sus predicciones. Cada árbol divide el espacio de
            variables en regiones y predice el promedio observado en esa
            región — por construcción, **nunca puede predecir un valor
            fuera del rango visto en entrenamiento** (no extrapola).

            ```yaml
            n_estimators: 300
            min_samples_leaf: 2
            ```

            **Como candidato a modelo de site-adaptation:** es de los más
            seguros para desplegar — por construcción nunca extrapola, así
            que siempre pasa el filtro de validez física. Su debilidad es
            que necesita bastantes horas de entrenamiento confiables para
            aprender patrones no lineales sin sobreajustar — con relativamente
            pocas horas supervisadas, una variante con aleatoriedad extra
            (Extra Trees) a veces lo supera por un margen pequeño.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, prueba):**")
            st.info("""
            RMSE = **98.66 W/m²** · MAE = **47.05 W/m²**
            0 predicciones bajo el mínimo físico simple (de 104.825); 4
            instantes (0.004%) quedan marcados por la prueba nocturna en
            pleno crepúsculo — dentro del margen de tolerancia justificado
            (ver metodología).

            Físicamente válido, pero a 0.06 W/m² del ganador (regresión
            lineal simple, RMSE 98.60): la diferencia es despreciable y
            muestra que la corrección es sobre todo de escala.
            """)

        st.markdown("**Cómo se ve — uno de los 300 árboles (ejemplo ilustrativo, "
                     "no ajustado a datos reales):**")
        st.plotly_chart(fig_diagrama_arbol_individual(), use_container_width=True, key="oe2_diagrama_arbol_rf")
        panel_info(
            "Cada árbol hace una serie de preguntas de sí/no sobre las "
            "variables predictoras (aquí, umbrales de ejemplo) hasta llegar a "
            "una hoja con un valor de predicción fijo — por construcción, ese "
            "valor siempre está dentro del rango visto en entrenamiento, nunca "
            "extrapola. Random Forest entrena 300 árboles como este, cada uno "
            "con una muestra aleatoria distinta de los datos, y promedia sus "
            "300 predicciones para dar el resultado final."
        )

    with st.expander("🌳 Modelo 6 — Extra Trees"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Cómo funciona:**")
            st.markdown("""
            Muy similar a Random Forest (bagging de árboles), pero en cada
            división **no busca el mejor corte posible** — sortea el corte
            entre varias opciones aleatorias. Esto añade aleatoriedad extra
            (de ahí "Extra" Trees), lo que puede reducir varianza cuando
            hay pocos datos de entrenamiento o el ruido es alto.

            **Como candidato a modelo de site-adaptation:** comparte la
            fortaleza de Random Forest (nunca extrapola, por construcción),
            con una ventaja adicional cuando el entrenamiento es más ruidoso
            o escaso — como en Udenar, donde gana por un margen muy pequeño
            (~0.2%) precisamente por esa aleatoriedad extra.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar 2023):**")
            st.info("""
            RMSE = **98.67 W/m²** (MAE = 47.05 W/m²), prácticamente igual a
            Random Forest (98.66) y a la regresión lineal simple (98.60, ganadora).

            0 predicciones bajo el mínimo físico simple; 3 de 104.825
            muestras del período completo (0.003%) quedan marcadas por la
            prueba nocturna, en pleno crepúsculo (elevación solar entre
            −0.02° y −0.41°) — un margen de tolerancia justificado y
            documentado, no una extrapolación real (ver metodología).
            """)

        st.markdown("**Cómo se ve — la diferencia clave frente a Random Forest:**")
        st.plotly_chart(fig_diagrama_extratrees_corte(), use_container_width=True, key="oe2_diagrama_extratrees")
        panel_info(
            "En cada nodo de cada árbol hay que decidir dónde \"cortar\" una "
            "variable para dividir los datos. Random Forest prueba todos los "
            "cortes posibles y elige el que más reduce el error. Extra Trees "
            "en cambio sortea el corte entre unas pocas opciones aleatorias — "
            "más rápido de entrenar y, con pocos datos, a veces generaliza "
            "mejor porque no se ajusta tanto al ruido de la muestra."
        )

    with st.expander("📊 Modelo 7 — Gradient Boosting"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Cómo funciona (boosting secuencial):**")
            st.markdown("""
            A diferencia de Random Forest/Extra Trees (árboles
            independientes en paralelo), Gradient Boosting entrena árboles
            **en secuencia**: cada árbol nuevo intenta corregir el error
            (residual) que dejó el árbol anterior. Puede capturar patrones
            muy finos, pero es más propenso a **sobreajustar** cuando hay
            relativamente pocas horas de entrenamiento (300-800 en este
            proyecto, frente a los miles/millones típicos donde boosting
            suele destacar).

            **Como candidato a modelo de site-adaptation:** es, en teoría,
            el más flexible de los tres modelos de árboles — pero también el
            que más entrenamiento necesita para aprovechar esa flexibilidad
            sin sobreajustar. En Udenar termina siendo el **mejor de los 9
            por RMSE puro**, pero queda descalificado por el filtro de
            validez física riguroso: al aplicarlo sobre TODO el período de
            aplicación (no solo el 30% de prueba), un porcentaje relevante
            de sus predicciones resulta físicamente imposible.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, prueba):**")
            st.info("""
            RMSE = **95.82 W/m²** · MAE = **46.80 W/m²** — el mejor RMSE
            puro de los candidatos, por delante incluso del ganador
            (regresión lineal simple, 98.60).

            Pero al aplicarlo sobre TODO el período de aplicación (104.825
            muestras), **13.223 predicciones (12.6%) son físicamente
            inválidas** (violan límites BSRN de negativo/PPL/ERL/nocturno)
            — muy por encima de cualquier margen de tolerancia razonable.
            En el 30% de prueba este problema no era visible; solo
            aparece al revisar el año completo. Por eso queda descalificado
            del despliegue pese a tener el mejor error: el filtro físico
            es un criterio de todo o nada, no un promedio, y debe aplicarse
            sobre todos los datos donde el modelo realmente se va a usar.
            """)

        st.markdown("**Cómo se ve — entrenamiento secuencial (esquema, 3 árboles):**")
        st.plotly_chart(fig_diagrama_gradient_boosting(), use_container_width=True,
                         key="oe2_diagrama_gradient_boosting")
        panel_info(
            "A diferencia de Random Forest/Extra Trees (árboles entrenados en "
            "paralelo e independientes), aquí cada árbol nuevo se entrena "
            "sobre lo que el árbol anterior dejó sin explicar (el residual). "
            "La predicción final es la suma de las contribuciones de todos "
            "los árboles. Esto le permite capturar patrones muy finos, pero "
            "con pocas horas de entrenamiento (300-800 en este proyecto) "
            "tiende a sobreajustar antes de que esa capacidad se aproveche."
        )

    with st.expander("🚀 Modelo 8 — XGBoost"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Cómo funciona (boosting regularizado):**")
            st.markdown("""
            Mismo principio que Gradient Boosting (árboles entrenados **en
            secuencia**, cada uno corrigiendo el residual del anterior),
            pero con regularización explícita en la función de pérdida
            (penaliza árboles muy complejos) y con submuestreo aleatorio de
            filas y columnas en cada árbol (`subsample`, `colsample_bytree`)
            — pensado para reducir el sobreajuste que afecta a Gradient
            Boosting con pocos datos de entrenamiento.

            ```yaml
            n_estimators: 300
            max_depth: 6
            learning_rate: 0.05
            subsample: 0.8
            colsample_bytree: 0.8
            ```

            **Como candidato a modelo de site-adaptation:** comparte el
            mismo riesgo estructural que Gradient Boosting — al ser
            boosting aditivo (suma de correcciones), no tiene ninguna
            garantía de quedarse dentro del rango de valores vistos en
            entrenamiento, a diferencia de Random Forest/Extra Trees. La
            regularización ayuda a que el problema sea menos severo que en
            Gradient Boosting, pero no lo elimina.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, prueba):**")
            st.info("""
            RMSE = **98,80 W/m²** · MAE = **47,98 W/m²** — rank 7 de 9,
            muy cerca de los árboles de bagging.

            Al aplicarlo sobre TODO el período de aplicación (104.825
            muestras), **1.345 predicciones (1,28%) resultan físicamente
            inválidas** — mucho menor que Gradient Boosting (15,5%) gracias
            a la regularización, pero sigue siendo un problema real: el
            filtro de validez física es de todo o nada, así que XGBoost
            también queda descalificado como ganador.
            """)

    with st.expander("🧠 Modelo 9 — Red neuronal (MLP)"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Cómo funciona (perceptrón multicapa):**")
            st.markdown("""
            Red neuronal densa con 2 capas ocultas (64 y 32 neuronas,
            activación ReLU), entrenada por descenso de gradiente (Adam)
            para minimizar el error cuadrático medio. A diferencia de los
            árboles, **es sensible a la escala de cada variable** — las
            predictoras se estandarizan (media 0, varianza 1) antes de
            entrenar, como parte del mismo pipeline del modelo.

            ```yaml
            hidden_layer_sizes: [64, 32]
            activation: relu
            solver: adam
            early_stopping: true
            ```

            **Como candidato a modelo de site-adaptation:** es, en teoría,
            el más flexible de los 9 candidatos — de hecho logra el mejor
            RMSE puro en Udenar-est. Pero es también, en la práctica, el
            que peor generaliza fuera de la ventana de entrenamiento/prueba:
            no tiene ninguna garantía estructural de mantenerse dentro del
            rango físico de GHI, igual que el boosting, pero el efecto
            resulta más severo.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, prueba):**")
            st.info("""
            RMSE = **98,18 W/m²** · MAE = **51,67 W/m²** — rank 2 de 9,
            mejor que los tres árboles de bagging y casi al nivel de
            Gradient Boosting.

            Pero al aplicarlo sobre TODO el período de aplicación,
            **43.248 predicciones (41,3%) resultan físicamente inválidas**
            — el peor resultado de los 9 candidatos, muy por encima incluso
            de Gradient Boosting (15,5%) o XGBoost (1,3%). Es el ejemplo
            más claro del proyecto de por qué el RMSE de prueba, solo, no
            alcanza para elegir un modelo de despliegue.
            """)

    st.markdown("---")
    subseccion("Resumen — modelo ganador por estación")

    _tablas_marco = cargar_reconstruccion_tablas()
    _df_met_marco = _tablas_marco["metricas"]
    if _df_met_marco is not None and not _df_met_marco.empty:
        _nombres_marco = {"udenar": "Udenar (terrestre)", CLAVE_UDENAR_EST: NOMBRE_UDENAR_EST}
        _df_ganadores = _df_met_marco[_df_met_marco["modelo_ganador"]].copy()
        _df_ganadores["Estación"] = _df_ganadores["estacion"].map(_nombres_marco)
        _df_ganadores_show = _df_ganadores[["Estación", "modelo", "RMSE", "MAE", "R2"]].copy()
        _df_ganadores_show.columns = ["Estación", "Modelo ganador", "RMSE [W/m²]", "MAE [W/m²]", "R²"]
        _df_ganadores_show["RMSE [W/m²]"] = _df_ganadores_show["RMSE [W/m²]"].round(2)
        _df_ganadores_show["MAE [W/m²]"] = _df_ganadores_show["MAE [W/m²]"].round(2)
        _df_ganadores_show["R²"] = _df_ganadores_show["R²"].round(4)
        tabla_html(_df_ganadores_show)
        boton_descarga_csv(_df_ganadores_show, "oe2_marco_modelos_ganadores.csv")

    panel_info(
        "Se descartan los modelos que producen "
        "valores físicamente imposibles en el período de aplicación; entre los "
        "modelos restantes, cuál gana depende de qué tan ruidosa sea la relación con "
        "Solcast y de cuántas horas confiables haya para entrenar: con una "
        "relación más ruidosa, EQM tiende a ganar; con más datos limpios, los "
        "árboles de bagging (Random Forest, Extra Trees) suelen capturar mejor "
        "los patrones no lineales sin sobreajustar; Gradient Boosting rara vez "
        "gana con volúmenes de datos moderados (cientos de horas). La tabla de "
        "arriba muestra cuál ganó realmente para la estación de la Universidad "
        "de Nariño."
    )


PRUEBAS_OE3_RECONSTRUCCION = {
    "neg_GHI": "GHI negativo",
    "noche_GHI": "GHI nocturno",
    "ppl_GHI": "Límite físico posible (PPL)",
    "erl_GHI": "Límite extremadamente raro (ERL)",
    "sensor_caido": "Sensor caído / plano",
    "csi": "Cielo claro (CSI)",
    "roc_GHI": "Tasa de cambio (RoC)",
    "persistencia_GHI": "Persistencia",
    "simetria_GHI": "Simetría AM/PM",
}


def render_oe3_marco_metodologico():
    """Marco metodológico de OE3: no introduce pruebas de QC ni métricas
    nuevas (reutiliza exactamente las de OE1 y las de Inicio), sino que
    explica el PROTOCOLO DE VERIFICACIÓN — cómo se combinan para validar
    la reconstrucción de OE2."""
    seccion("Marco metodológico — Protocolo de verificación de desempeño")

    st.markdown("""
    Esta etapa **no define pruebas de QC ni métricas nuevas** — reutiliza
    exactamente las pruebas físicas/expertas de la etapa de control de
    calidad (ver su marco metodológico) y las métricas de error/usabilidad
    de Inicio (ver su marco metodológico). Lo que aporta esta etapa es el
    **protocolo de verificación**: la forma específica de combinarlas para
    responder una pregunta que la corrección por sí sola no puede contestar:

    > *¿La mejora estadística de la reconstrucción (menor RMSE) es también
    > una mejora físicamente válida, o el modelo simplemente "suaviza" el
    > error sin corregir el problema de fondo?*
    """)

    with st.expander("🔁 Paso 1 — Re-aplicar el mismo QC a 3 versiones", expanded=True):
        st.markdown("""
        Se corre exactamente el mismo control de calidad de la etapa inicial
        (mismas pruebas, mismos umbrales) sobre **3 versiones** de cada
        estación terrestre, sobre todo el período de aplicación:

        1. **Crudo** — el dato original de la estación, sin tocar.
        2. **Corregido — total** — la predicción del modelo en cada hora.
        3. **Corregido — selectivo** — medición real donde ya era válida,
           predicción del modelo en lo flageado/faltante.

        Usar el **mismo** control de calidad (no una versión relajada) es lo
        que hace la comparación honesta: si el dato reconstruido tuviera
        problemas físicos nuevos, esta prueba los mostraría igual que
        mostró los problemas del dato crudo en la etapa inicial.
        """)

    with st.expander("📊 Paso 2 — Comparar % usable (Familia 2 de Inicio)"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("""
            Se compara el `% usable` (ver marco metodológico de Inicio) de
            las 3 versiones. Un aumento real de `% usable` después de
            corregir es la evidencia más directa de que el problema physical
            (no solo el numérico) mejoró.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar):**")
            st.info("""
            Crudo: **99.85%** usable
            Corregido (total y selectivo): **100.0%** usable (+0.15 pp)

            El crudo ya partía con un %usable muy alto (pocas muestras
            fallaban el QC físico/experto de entrada), así que la
            reconstrucción tenía poco margen de mejora — aun así sube
            levemente, no baja. Es consistente con que el modelo ganador
            (regresión lineal simple) se eligió exactamente con el filtro de validez
            física riguroso sobre todo el período de aplicación: por
            diseño, casi no introduce muestras físicamente inválidas
            nuevas. El Paso 4 (desglose por prueba) muestra el detalle.
            """)

    with st.expander("🔬 Paso 3 — Comparar métricas de error (Familia 1 de Inicio)"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("""
            Se comparan MAE/RMSE/R² del modelo ganador contra el baseline
            (Solcast sin corregir), **siempre sobre el conjunto de prueba
            honesto** (30% final, nunca visto en entrenamiento) — nunca
            sobre el período de aplicación completo, que mezclaría datos
            usados para entrenar.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, prueba):**")
            st.info("""
            Baseline: MAE 64.54 · RMSE 123.64 · R² 0.690
            Udenar adaptado (regresión lineal simple): MAE 48.21 · RMSE 98.60 · R² 0.803
            """)

    with st.expander("🔎 Paso 4 — Detalle de banderas por prueba (detectar artefactos)"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("""
            El `% usable` es un promedio — puede ocultar que el modelo
            arregla un problema pero introduce otro distinto. Por eso se
            desglosa el % de banderas **por cada prueba individual**
            (nocturno, límites físicos, persistencia, CSI, etc.): esto
            permite ver, prueba por prueba, si la reconstrucción realmente
            mejora el dato o simplemente mueve el problema — incluyendo el
            caso, honesto y menos cómodo, de que alguna prueba individual
            **empeore** tras la corrección.
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, crudo → corregido total):**")
            st.info("""
            GHI nocturno: **0.048% → 0.003%** (mejora)
            Cielo claro (CSI): **0.102% → 0.0%** (mejora)
            Simetría AM/PM: **11.65% → 22.48%** (empeora, ~2×)

            El resto de las pruebas (negativo, límites físicos PPL/ERL,
            persistencia, tasa de cambio) están en 0% de banderas en ambas
            versiones — no hay artefacto ahí. Nocturno y CSI incluso
            mejoran: el modelo, entrenado y filtrado con el criterio
            riguroso sobre todo el período, casi no introduce violaciones
            físicas nuevas. La excepción clara es simetría AM/PM, que sí
            empeora notablemente al reemplazar el dato completo — es una
            prueba diagnóstica (no cuenta en el `% usable`, ver marco
            metodológico de Control de Calidad), pero el detalle deja
            constancia de que el modelo altera la forma AM/PM de la curva
            diaria aunque no afecte la validez física básica.
            """)

    panel_info(
        "En resumen: esta etapa valida la reconstrucción de site-adaptation "
        "con la misma vara usada en el control de calidad inicial (las "
        "pruebas de QC) y en Inicio (las métricas de "
        "error/usabilidad) — nunca con un criterio nuevo o más permisivo. "
        "En Udenar esa vara honesta muestra un resultado consistente: el "
        "modelo gana claramente en error de predicción (RMSE, MAE, R²) "
        "frente al baseline, y el `% usable` y el detalle de banderas por "
        "prueba también mejoran o se mantienen — con la única excepción de "
        "simetría AM/PM (prueba diagnóstica, no cuenta en el `% usable`), "
        "que empeora de forma notable. Una mejora estadística no siempre "
        "se traduce en una mejora física completa, y por eso el protocolo "
        "verifica ambas por separado en vez de asumir que una implica la "
        "otra."
    )


_ESTACIONES_OE3 = {
    "Udenar (terrestre)": ("udenar", C_RED),
    NOMBRE_UDENAR_EST: (CLAVE_UDENAR_EST, C_UDENAR_EST),
}


def render_oe3_evolucion_completa():
    """Pestaña de OE3: evolución completa del dato de la estación terrestre,
    fuente satelital (referencia) -> crudo -> QC OE1 -> adaptado OE2 -> re-QC
    OE3. Muestra que el dataset definitivo (adaptado + re-QC) es un
    subconjunto distinto del adaptado de OE2, porque el re-QC descarta
    algunos puntos adaptados."""
    seccion("Evolución completa del dato — de Solcast a la serie final")

    tooltip("""
    **¿Por qué esta gráfica es distinta a la de site-adaptation?**

    En la etapa de site-adaptation, la serie "adaptada" muestra el 100% de
    los puntos que produce el modelo — nunca se descarta ninguno ahí. Pero
    en esta etapa de verificación se le vuelve a aplicar el **mismo control
    de calidad de la etapa inicial** a esa serie adaptada, y algunos puntos
    (los mismos que dispararían una bandera si fueran datos crudos) quedan
    marcados como no usables. Es decir: el dataset "definitivo" (adaptado +
    re-control de calidad) es un subconjunto más pequeño que el "adaptado"
    de la etapa de site-adaptation, no el mismo dataset.

    Esta gráfica muestra las 5 etapas superpuestas para cada estación:

    1. **Solcast** (área sombreada gris, dato a resolución nativa de 5 min
       tal cual — sin promediar) — referencia de fondo, para ubicar
       visualmente el nivel y el ciclo día/noche esperado.
    2. **Crudo (100%)** — el dato original de la estación, sin filtrar.
    3. **Crudo que pasa control de calidad** — subconjunto que aprueba las
       pruebas físicas/expertas (mismo criterio que la pestaña "Resumen
       comparativo" de la etapa inicial).
    4. **Adaptado (100%)** — la predicción del modelo aplicada a todo el
       período (línea punteada, el dataset completo de site-adaptation,
       sin filtrar).
    5. **Adaptado que pasa el re-control de calidad** — subconjunto de (4)
       que además aprueba el mismo control de calidad de la etapa inicial —
       es el dataset final, físicamente verificado.
    """)

    col_sel1, col_sel2 = st.columns([1.3, 1])
    with col_sel1:
        _fuente_evol_oe3 = st.selectbox(
            "Estación:", list(_ESTACIONES_OE3.keys()), key="oe3_evol_fuente")
    with col_sel2:
        _variante_evol_oe3 = st.radio(
            "Variante adaptada:", ["total", "selectivo"],
            horizontal=True, key="oe3_evol_variante",
            format_func=lambda v: "Total" if v == "total" else "Selectivo",
        )

    _CAPAS_OE3_LABELS = {
        "solcast": "☁️ Solcast (sombra)",
        "crudo": "🔴 Crudo (100%)",
        "crudo_qc": "🟢 Crudo que pasa control de calidad",
        "adapt_100": "🟡 Adaptado (100%)",
        "adapt_qc": "🔵 Adaptado que pasa el re-control de calidad",
    }
    _capas_sel_oe3 = st.multiselect(
        "Capas a mostrar (quita o pon las que quieras ver):",
        options=list(_CAPAS_OE3_LABELS.keys()),
        default=list(_CAPAS_OE3_LABELS.keys()),
        format_func=lambda k: _CAPAS_OE3_LABELS[k],
        key="oe3_evol_capas",
    )

    _clave_evol_oe3, _color_evol_oe3 = _ESTACIONES_OE3[_fuente_evol_oe3]

    if not _capas_sel_oe3:
        st.warning("Selecciona al menos una capa para dibujar la gráfica.")
        return

    _fig_evol_oe3 = fig_evolucion_reconstruccion_plotly(
        _clave_evol_oe3, _fuente_evol_oe3, _color_evol_oe3, variante=_variante_evol_oe3,
        capas=tuple(_capas_sel_oe3))

    if _fig_evol_oe3 is not None:
        st.plotly_chart(_fig_evol_oe3, use_container_width=True,
                         key=f"oe3_evol_completa_{_clave_evol_oe3}_{_variante_evol_oe3}")
        panel_info(
            f"En {_fuente_evol_oe3}, comparar la capa 4 (adaptado, 100%) contra la "
            f"capa 5 (adaptado que pasa el re-QC) muestra exactamente cuántos puntos "
            f"del dataset adaptado se descartan al volver a aplicar el control de "
            f"calidad en la verificación final — esa diferencia es la que separa el "
            f"'% usable corregido' final del 100% que muestra la reconstrucción."
        )
    else:
        st.info(f"No hay datos suficientes para {_fuente_evol_oe3}. Corre el paso de "
                f"reconstrucción y confirma que existan los archivos de control de "
                f"calidad y de corrección para esta estación.")


def _fig_cierre_radiativo_solcast(df_solcast: "pd.DataFrame | None") -> "go.Figure | None":
    """Figura de cierre radiativo de la fuente satelital, coloreada por flag_cierre.

    Compara GHI medido vs. GHI calculado (DNI·cos(SZA)+DHI); se expone como
    función para reutilizarla en el mosaico de diagnóstico."""
    if df_solcast is None or not {"GHI", "DNI", "DHI", "zenith"}.issubset(df_solcast.columns):
        return None
    d = df_solcast[df_solcast["GHI"] > 50].copy()
    if d.empty:
        return None
    d["GHI_calc"] = d["DNI"] * np.cos(np.radians(d["zenith"])) + d["DHI"]
    flag_col = "flag_cierre" if "flag_cierre" in d.columns else None
    fig = go.Figure()
    if flag_col:
        for et, color in [("OK", "#22AA44"), ("FLAG", "#E63946")]:
            sub = d[d[flag_col] == et]
            if len(sub) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=sub["GHI_calc"], y=sub["GHI"], mode="markers",
                marker=dict(size=3, color=color, opacity=0.5),
                name=f"{et} ({len(sub):,})",
                hovertemplate="Calculado: %{x:.0f} W/m²<br>Medido: %{y:.0f} W/m²<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(x=d["GHI_calc"], y=d["GHI"], mode="markers",
                                  marker=dict(size=3, color=COLORES_FUENTES["Solcast"], opacity=0.5), name="Solcast"))
    lim = float(max(d["GHI_calc"].max(), d["GHI"].max())) * 1.02
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                              line=dict(color="black", dash="dash", width=1.2), name="Cierre perfecto (1:1)"))
    fig.update_layout(
        title=dict(text="Solcast — cierre radiativo: GHI medido vs. calculado", font=dict(size=13)),
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=40, b=20, l=50, r=10),
        legend=dict(orientation="h", y=1.1, x=0),
        xaxis=dict(title="DNI·cos(SZA) + DHI [W/m²]", gridcolor="#f0f0f0"),
        yaxis=dict(title="GHI medido [W/m²]", gridcolor="#f0f0f0"),
    )
    return fig


def render_oe3_mosaico_diagnostico():
    """Mosaico de diagnóstico por estación, inspirado en el multiplot de
    referencia para QC experto de radiación solar (solar_multiplot).

    Muestra, para la estación seleccionada, los 4 diagnósticos que SÍ
    dependen de esa estación (CSI, simetría AM/PM, patrón anual día × hora,
    envolvente), en sus 4 etapas de depuración: crudo / post-QC inicial /
    site-adaptado / post re-QC final (ver `_construir_df_etapa_diagnostico`).

    Los diagnósticos que solo se pueden calcular con las 3 componentes
    (GHI+DNI+DHI: cierre radiativo, K-tests, sesgo DNI vs. azimut, patrones
    de DNI) no se incluyen: solo la fuente satelital dispone de las tres
    componentes, así que no dependen de la estación seleccionada y se muestran
    en la pestaña de la fuente satelital de la etapa de control de calidad."""
    seccion("Mosaico de diagnóstico visual")

    tooltip("""
    **¿Qué es este mosaico?**

    Es una batería estándar de gráficas de diagnóstico visual usada en
    control de calidad de radiación solar, comprimida en un solo bloque
    compacto para poder ver de un vistazo el panorama completo de una
    estación en sus 4 etapas: ① crudo (sin filtrar), ② post-QC inicial
    (control de calidad físico + experto), ③ site-adaptado (resultado
    directo de esa etapa, antes del re-QC final) y ④ post re-QC (el
    dataset definitivo).

    Los diagnósticos que solo se pueden calcular con las 3 componentes
    (GHI+DNI+DHI) — cierre radiativo, K-tests, sesgo DNI vs. azimut,
    patrones de DNI — no aparecen aquí: solo Solcast los tiene, así que no
    cambian según la estación seleccionada. Ya están disponibles en la
    pestaña **"Solcast"** de la etapa de control de calidad.
    """)

    _fuente_mos = st.selectbox("Estación terrestre a mostrar junto a Solcast:",
                                list(_ESTACIONES_OE3.keys()), key="oe3_mosaico_fuente")
    _clave_mos, _ = _ESTACIONES_OE3[_fuente_mos]

    _df_sa_mos = _construir_df_etapa_diagnostico(_clave_mos, "site_adapt")
    _dfs4_mos = (
        _construir_df_etapa_diagnostico(_clave_mos, "crudo"),
        _construir_df_etapa_diagnostico(_clave_mos, "post_qc"),
        _df_sa_mos,
        _construir_df_etapa_diagnostico(_clave_mos, "post_reqc"),
    )
    _etq4_mos = ("① Crudo", "② Post-QC", "③ Site-adaptado", "④ Post re-QC")

    # --- Diagnóstico compacto de la estación, las 4 etapas, en un solo pantallazo ---
    subseccion(f"{_fuente_mos} — diagnóstico compacto (4 etapas: crudo / post-QC / "
               "site-adaptado / post re-QC)")
    tooltip(
        "Los mismos 4 diagnósticos ya usados en la etapa de site-adaptation, "
        "comprimidos aquí en un solo bloque para verlos todos juntos: CSI y "
        "simetría AM/PM arriba (una gráfica con las 4 series cada una), patrón "
        "anual y envolvente abajo (un panel por etapa, escala de color "
        "compartida). Quedan más pequeños a propósito, para caber en una sola "
        "pantalla — usa el ícono de expandir (⛶) de cada gráfica para verla "
        "en grande."
    )
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        _fig_csi4_mos = fig_csi_comparativo_n_plotly(_dfs4_mos, _etq4_mos, _fuente_mos)
        if _fig_csi4_mos is not None:
            st.plotly_chart(_fig_csi4_mos, use_container_width=True, key=f"mosaico_csi4_{_clave_mos}")
        else:
            st.info(f"No hay datos suficientes de {_fuente_mos} para CSI.")
    with col_a2:
        _fig_sim4_mos = fig_simetria_comparativo_n_plotly(_dfs4_mos, _etq4_mos, _fuente_mos)
        if _fig_sim4_mos is not None:
            st.plotly_chart(_fig_sim4_mos, use_container_width=True, key=f"mosaico_simetria4_{_clave_mos}")
        else:
            st.info(f"No hay datos suficientes de {_fuente_mos} para simetría AM/PM.")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        _fig_hm4_mos = fig_heatmap_comparativo_plotly(
            _dfs4_mos, _etq4_mos, "GHI", _fuente_mos, alto_por_fila=115)
        if _fig_hm4_mos is not None:
            st.plotly_chart(_fig_hm4_mos, use_container_width=True, key=f"mosaico_heatmap4_{_clave_mos}")
        else:
            st.info(f"No hay datos suficientes de {_fuente_mos} para el patrón anual.")
    with col_b2:
        _fig_env4_mos = fig_envolvente_comparativo_plotly(
            _dfs4_mos, _etq4_mos, "GHI", _fuente_mos, alto_por_fila=115)
        if _fig_env4_mos is not None:
            st.plotly_chart(_fig_env4_mos, use_container_width=True, key=f"mosaico_envolvente4_{_clave_mos}")
        else:
            st.info(f"No hay datos suficientes de {_fuente_mos} para la envolvente.")
    panel_info(
        "① crudo, ② post-QC, ③ site-adaptado (resultado de esta etapa, sin el re-QC "
        "final) y ④ post re-QC (dataset definitivo). Si cada etapa mejora la calidad "
        "del dato, se debería ver una progresión ①→②→③→④ hacia CSI más centrado en 1, "
        "menos huecos en el patrón anual y envolventes más consistentes entre sí."
    )

    # --- Dispersión 1:1 y ECDF (mismas gráficas de OE2, para ver el desempeño
    # del modelo ganador junto al resto del diagnóstico) ---
    subseccion(f"{_fuente_mos} — desempeño del modelo ganador (antes/después del site-adaptation)")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        _fig_disp_mos, _stats_disp_mos = fig_dispersion_1a1_plotly(_clave_mos, _fuente_mos)
        if _fig_disp_mos is not None:
            st.plotly_chart(_fig_disp_mos, use_container_width=True, key=f"mosaico_dispersion_{_clave_mos}")
        else:
            st.info(f"No hay predicciones de prueba disponibles todavía para {_fuente_mos}.")
    with col_c2:
        _fig_ecdf_mos = fig_ecdf_plotly(_clave_mos, _fuente_mos)
        if _fig_ecdf_mos is not None:
            st.plotly_chart(_fig_ecdf_mos, use_container_width=True, key=f"mosaico_ecdf_{_clave_mos}")
        else:
            st.info(f"No hay predicciones de prueba disponibles todavía para {_fuente_mos}.")
    if _stats_disp_mos is not None:
        panel_info(
            "Dispersión: cada punto es un instante del conjunto de prueba; mientras más "
            "pegados a la diagonal 1:1, mejor. Antes de la corrección la pendiente del "
            f"ajuste es {_stats_disp_mos['m_antes']:.2f} ({_stats_disp_mos['ang_antes']:.0f}°), "
            f"y después sube a {_stats_disp_mos['m_desp']:.2f} ({_stats_disp_mos['ang_desp']:.0f}°) "
            "— más cerca del caso ideal (45°/pendiente=1). "
            "ECDF: si el site-adaptation funciona bien, la curva azul (adaptado) debe quedar "
            "prácticamente encima de la negra (medido); la separación de la roja (Solcast sin "
            "corregir) es la distorsión de distribución que el modelo corrige."
        )


def render_tab_oe3_estacion(clave: str, nombre_display: str, color: str):
    """Contenido de una pestaña de estación en OE3: re-QC crudo vs. corregido
    (2 variantes), métricas estadísticas baseline vs. modelo ganador, y
    detalle de flags por prueba."""
    tablas = cargar_reconstruccion_tablas()
    df_resumen, df_metricas, df_comp = tablas["resumen"], tablas["metricas"], tablas["comparativo"]

    seccion(f"{nombre_display} — re-control de calidad: crudo vs. corregido", color)

    _dfm_est = df_metricas[df_metricas["estacion"] == clave] if df_metricas is not None else None
    _baseline = None
    _ganador_row = None
    if _dfm_est is not None and not _dfm_est.empty:
        _base_sel = _dfm_est[_dfm_est["modelo"].str.contains("Baseline", case=False, na=False)]
        if not _base_sel.empty:
            _baseline = _base_sel.iloc[0]
        _gan_sel = _dfm_est[_dfm_est["modelo_ganador"]]
        if not _gan_sel.empty:
            _ganador_row = _gan_sel.iloc[0]

    if _baseline is not None and _ganador_row is not None:
        _nombre_modelo = _ganador_row["modelo"]

        def _mejora_pct(antes, despues, menor_es_mejor=True):
            """Mejora porcentual entre dos valores de una métrica.

            Si menor_es_mejor es True, una reducción cuenta como mejora.
            Devuelve None si antes es 0 o el cálculo falla."""
            try:
                if antes == 0:
                    return None
                if menor_es_mejor:
                    return 100 * (antes - despues) / abs(antes)
                return 100 * (despues - antes) / abs(antes)
            except Exception:
                return None

        _m_mae = _mejora_pct(_baseline["MAE"], _ganador_row["MAE"])
        _m_rmse = _mejora_pct(_baseline["RMSE"], _ganador_row["RMSE"])
        _m_rrmse = _mejora_pct(_baseline["rRMSE"], _ganador_row["rRMSE"])
        _m_r2 = _mejora_pct(_baseline["R2"], _ganador_row["R2"], menor_es_mejor=False)

        subseccion(f"Desempeño del modelo — evaluado contra GHI real de {nombre_display} "
                   f"(conjunto de prueba, modelo: {_nombre_modelo})")
        tooltip(f"""
        **¿Contra qué se compara cada columna?**

        Las dos columnas de la tabla de abajo (y las 4 métricas de arriba) se
        evalúan contra **la misma referencia: el GHI real medido en
        {nombre_display}**, en las horas de prueba donde esa medición es
        confiable (30% final del conjunto supervisado, nunca visto en
        entrenamiento).

        - **"Solcast sin corregir"** = usar el dato de Solcast tal cual como
          si fuera el GHI de {nombre_display} — es el punto de partida que
          se reemplaza, **no** la referencia.
        - **"Corregido ({_nombre_modelo})"** = la predicción del modelo
          ganador, también evaluada contra ese mismo GHI real de
          {nombre_display}.

        No existe una fila "{nombre_display} crudo" en esta tabla porque
        el GHI real de {nombre_display} **es** la referencia contra la que
        se miden ambas columnas — comparar {nombre_display} contra sí misma
        daría error cero por definición. La pregunta que esta tabla
        contesta es otra: *si tuviera que estimar el GHI de {nombre_display}
        sin usar su propio sensor, ¿me conviene más el dato crudo de
        Solcast o el dato ya corregido?*

        La comparación **física** (crudo de {nombre_display} vs. corregido,
        sin usar Solcast como referencia) es la sección de abajo
        ("% de datos utilizables") — ahí sí se evalúa la serie cruda de
        {nombre_display} en sí misma, con las mismas pruebas de control de
        calidad de la etapa inicial.
        """)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{_ganador_row['MAE']:.1f} W/m²",
                  delta=(f"−{_m_mae:.1f}%" if _m_mae is not None else "N/D"), delta_color="normal")
        c2.metric("RMSE", f"{_ganador_row['RMSE']:.1f} W/m²",
                  delta=(f"−{_m_rmse:.1f}%" if _m_rmse is not None else "N/D"), delta_color="normal")
        c3.metric("rRMSE", f"{_ganador_row['rRMSE']:.1f}%",
                  delta=(f"−{_m_rrmse:.1f}%" if _m_rrmse is not None else "N/D"), delta_color="normal")
        c4.metric("R²", f"{_ganador_row['R2']:.3f}",
                  delta=(f"+{_m_r2:.1f}%" if _m_r2 is not None else "N/D"), delta_color="normal")

        _tabla_met = pd.DataFrame({
            "Métrica": ["MAE [W/m²]", "RMSE [W/m²]", "MBE [W/m²]", "rRMSE [%]", "Pearson R", "R²"],
            "Solcast sin corregir (baseline)": [
                f"{_baseline['MAE']:.2f}", f"{_baseline['RMSE']:.2f}", f"{_baseline['MBE']:+.2f}",
                f"{_baseline['rRMSE']:.2f}", f"{_baseline['Pearson_R']:.4f}", f"{_baseline['R2']:.4f}",
            ],
            f"Corregido ({_nombre_modelo})": [
                f"{_ganador_row['MAE']:.2f}", f"{_ganador_row['RMSE']:.2f}", f"{_ganador_row['MBE']:+.2f}",
                f"{_ganador_row['rRMSE']:.2f}", f"{_ganador_row['Pearson_R']:.4f}", f"{_ganador_row['R2']:.4f}",
            ],
        })
        st.caption(f"Ambas columnas evaluadas contra el GHI real medido de {nombre_display} "
                   f"(conjunto de prueba) — nunca una contra la otra.")
        tabla_html(_tabla_met)
        boton_descarga_csv(_tabla_met, f"oe3_{clave}_metricas_comparativas.csv")
    else:
        st.info(f"No hay métricas de modelos para {nombre_display}. "
                f"Corre el paso de entrenamiento y evaluación de los modelos de site-adaptation.")

    # ── Dispersión GHI real vs. satelital crudo / corregido (dataset definitivo) ──
    _df_disp = cargar_procesado(f"{clave}_ghi_corregido.csv")
    _df_sol_disp = cargar_procesado(f"{SOLCAST_TAG_POR_FUENTE.get(clave, 'solcast')}_qc_completo.csv")
    if _df_disp is not None and _df_sol_disp is not None and "origen_dato" in _df_disp.columns:
        subseccion("Visualizaciones de desempeño (dataset definitivo, post re-QC)")
        _variante_disp = st.radio(
            "Variante corregida a graficar:", ["total", "selectivo"],
            horizontal=True, key=f"oe3_disp_variante_{clave}",
            format_func=lambda v: "Total" if v == "total" else "Selectivo",
        )
        _col_corr_disp = f"GHI_corregido_{_variante_disp}"
        _col_usab_disp = f"usabilidad_{_variante_disp}"

        _sol_merge = _df_sol_disp[["datetime", "GHI"]].rename(columns={"GHI": "GHI_solcast"})
        _dfd = _df_disp.merge(_sol_merge, on="datetime", how="inner")
        _dfd_real = _dfd[_dfd["origen_dato"] == "medido_ok"].dropna(subset=["GHI_original", "GHI_solcast"])

        if not _dfd_real.empty:
            _dfd_final = _dfd_real[_dfd_real[_col_usab_disp] == "OK"].dropna(subset=[_col_corr_disp])

            _lim_disp = float(max(
                _dfd_real["GHI_original"].max(),
                _dfd_real["GHI_solcast"].max(),
                _dfd_final[_col_corr_disp].max() if not _dfd_final.empty else 0,
            )) * 1.05

            _fig_disp = make_subplots(rows=1, cols=2, subplot_titles=(
                f"Solcast crudo vs {nombre_display}",
                f"Corregido ({_variante_disp}, post re-QC) vs {nombre_display}"))
            _fig_disp.add_trace(go.Scatter(
                x=_dfd_real["GHI_original"], y=_dfd_real["GHI_solcast"],
                mode="markers", marker=dict(color=COLORES_FUENTES["Solcast"], size=4, opacity=0.4),
                name="Solcast crudo", showlegend=False,
            ), row=1, col=1)
            _fig_disp.add_trace(go.Scatter(
                x=[0, _lim_disp], y=[0, _lim_disp], mode="lines",
                line=dict(color="black", dash="dash", width=1), name="1:1", showlegend=False,
            ), row=1, col=1)
            _fig_disp.add_trace(go.Scatter(
                x=_dfd_final["GHI_original"], y=_dfd_final[_col_corr_disp],
                mode="markers", marker=dict(color=color, size=4, opacity=0.4),
                name="Corregido", showlegend=False,
            ), row=1, col=2)
            _fig_disp.add_trace(go.Scatter(
                x=[0, _lim_disp], y=[0, _lim_disp], mode="lines",
                line=dict(color="black", dash="dash", width=1), name="1:1", showlegend=False,
            ), row=1, col=2)
            _fig_disp.update_xaxes(title_text=f"GHI {nombre_display} [W/m²]", gridcolor="#f0f0f0")
            _fig_disp.update_yaxes(title_text="GHI predicho [W/m²]", gridcolor="#f0f0f0", col=1)
            _fig_disp.update_layout(
                height=440, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40, b=20, l=60, r=20),
            )
            st.plotly_chart(_fig_disp, use_container_width=True, key=f"oe3_disp_{clave}_{_variante_disp}")
            panel_info(
                f"Dispersión GHI sobre TODAS las horas con medición confiable de {nombre_display} "
                f"(origen_dato == medido_ok, {len(_dfd_real):,} horas): Solcast crudo (izq.) vs. "
                f"corregido ya verificado por el re-control de calidad (der., variante {_variante_disp}, "
                f"{len(_dfd_final):,} horas que sobreviven el re-QC). A diferencia de la vista del "
                f"Anexo (que solo usa el 30% de prueba, ~802h de Alvernia), esta compara contra el "
                f"año completo de mediciones confiables de {nombre_display}."
            )
        else:
            st.info(f"No hay horas con origen_dato == 'medido_ok' para {nombre_display}.")

    if df_comp is not None:
        _dfc = df_comp[df_comp["estacion"] == clave]
        if not _dfc.empty and df_resumen is not None and clave in df_resumen["estacion"].values:
            subseccion("% de datos utilizables (re-QC) — crudo vs. corregido")
            st.caption(f"Aquí sí se evalúa la serie cruda de {nombre_display} en sí misma "
                       f"(sobre TODO el período de aplicación, no solo el conjunto de prueba), "
                       f"con las mismas pruebas físicas/expertas de la etapa inicial — es la comparación "
                       f"'antes vs. después' que buscas ver junto a la tabla de arriba.")
            _n_total = df_resumen[df_resumen["estacion"] == clave].iloc[0]["muestras_aplicacion_total"]
            _pct_crudo = _dfc["pct_usable_crudo"].iloc[0]
            _fila_total = _dfc[_dfc["variante"] == "total"]
            _fila_sel = _dfc[_dfc["variante"] == "selectivo"]
            _pct_total = _fila_total["pct_usable_corregido"].iloc[0] if not _fila_total.empty else float("nan")
            _pct_sel = _fila_sel["pct_usable_corregido"].iloc[0] if not _fila_sel.empty else float("nan")

            col_a, col_b, col_c = st.columns(3)
            for _col, _label, _pct, _c in [
                (col_a, "Crudo (sin corregir)", _pct_crudo, "#4d4d4d"),
                (col_b, "Corregido — total", _pct_total, COLOR_CORREGIDO_TOTAL),
                (col_c, "Corregido — selectivo", _pct_sel, COLOR_CORREGIDO_SELECTIVO),
            ]:
                _col.markdown(
                    f"<div style='background:{C_LIGHT}; border-radius:12px; padding:18px; "
                    f"border-left:5px solid {_c}'>"
                    f"<b style='font-size:0.95rem; color:{_c}'>{_label}</b><br>"
                    f"<span style='font-size:2.2rem; font-weight:700; color:{_c}'>{_pct:.2f}%</span>"
                    f"<br><small style='color:#666'>{_pct/100*_n_total:,.0f} de {_n_total:,.0f} horas</small>"
                    f"</div>", unsafe_allow_html=True
                )

    _df_flags = cargar_csv(f"reconstruccion_{clave}_flags_comparativo.csv")
    if _df_flags is not None and not _df_flags.empty:
        subseccion("Detalle de flags por prueba — crudo vs. corregido (total / selectivo)")
        _dff = _df_flags.copy()
        _dff["prueba"] = _dff["prueba"].map(lambda x: PRUEBAS_OE3_RECONSTRUCCION.get(x, x))
        _pivot = _dff.pivot_table(index="prueba", columns="serie", values="pct_flag", aggfunc="first").reset_index()
        _pivot.columns.name = None
        _orden_cols = ["prueba"] + [c for c in ["crudo", "total", "selectivo"] if c in _pivot.columns]
        _pivot = _pivot[_orden_cols]
        _rename_cols = {"prueba": "Prueba", "crudo": "Crudo", "total": "Corregido (total)",
                        "selectivo": "Corregido (selectivo)"}
        _pivot = _pivot.rename(columns=_rename_cols)
        _cols_pct = [c for c in _pivot.columns if c != "Prueba"]
        tabla_html(_pivot, formato={c: "{:.3f}%" for c in _cols_pct})
        boton_descarga_csv(_pivot, f"oe3_{clave}_flags_comparativo.csv")

        tooltip(f"""
        **¿Cómo leer esta tabla?**

        Cada fila es una prueba de control de calidad (idéntica a la usada en la etapa inicial);
        cada columna, el % de horas de {nombre_display} que dispara esa bandera en
        esa versión de la serie. Una reconstrucción exitosa reduce el % de banderas
        en casi todas las pruebas frente al crudo — especialmente en las pruebas que
        capturan huecos y errores de sensor (nocturno, límites físicos, persistencia),
        que es justamente lo que Solcast como referencia permite corregir y rellenar.
        """)
    else:
        st.info(f"No hay detalle de flags para {nombre_display}. "
                f"Corre el paso de reconstrucción para generar esta comparación.")

    # ── Diagnóstico visual — 4 etapas: crudo / post-QC / site-adaptado / post re-QC ──
    _dg4_crudo = _construir_df_etapa_diagnostico(clave, "crudo")
    _dg4_postqc = _construir_df_etapa_diagnostico(clave, "post_qc")
    _dg4_sa = _construir_df_etapa_diagnostico(clave, "site_adapt")
    _dg4_reqc = _construir_df_etapa_diagnostico(clave, "post_reqc")
    _dg4_dfs = (_dg4_crudo, _dg4_postqc, _dg4_sa, _dg4_reqc)
    _dg4_etq = ("① Crudo", "② Post-QC", "③ Site-adaptado", "④ Post re-QC")

    if any(d is not None for d in _dg4_dfs):
        subseccion(f"Diagnóstico visual — {nombre_display}: las 4 etapas")
        tooltip(
            "Misma metodología ya usada en la etapa de site-adaptation (CSI, simetría "
            "AM/PM, patrón anual, envolvente), agregando ahora una 4ta etapa: **④ post "
            "re-QC**, el dataset definitivo de esta pestaña (③ pasado otra vez por el "
            "control de calidad) — así se ve de un vistazo si cada etapa efectivamente "
            "mejora estos diagnósticos frente a la anterior."
        )

        _fig_csi4 = fig_csi_comparativo_n_plotly(_dg4_dfs, _dg4_etq, nombre_display)
        if _fig_csi4 is not None:
            st.plotly_chart(_fig_csi4, use_container_width=True, key=f"csi_cmp4_{clave}")
            panel_info(
                "CSI diario mediano en las 4 etapas — si cada etapa mejora la calidad del "
                "dato, las curvas deberían verse progresivamente más centradas alrededor de "
                "1 y con menos picos extremos, en el orden ①→②→③→④."
            )

        _fig_sim4 = fig_simetria_comparativo_n_plotly(_dg4_dfs, _dg4_etq, nombre_display)
        if _fig_sim4 is not None:
            st.plotly_chart(_fig_sim4, use_container_width=True, key=f"simetria_cmp4_{clave}")
            panel_info(
                "Asimetría AM/PM en las 4 etapas — una asimetría sistemática que se "
                "mantiene igual en las 4 series es evidencia de un forzante climático "
                "real, no un artefacto que el QC, el site-adaptation o el re-QC deberían "
                "corregir."
            )

        subseccion(f"Patrón anual día × hora — {nombre_display}: las 4 etapas")
        _fig_hm4 = fig_heatmap_comparativo_plotly(_dg4_dfs, _dg4_etq, "GHI", nombre_display)
        if _fig_hm4 is not None:
            st.plotly_chart(_fig_hm4, use_container_width=True, key=f"heatmap_cmp4_{clave}")
            panel_info(
                "Mapa de calor día del año × hora local, con las 4 etapas compartiendo la "
                "MISMA escala de color. ③ y ④ deberían verse con celdas más uniformes que "
                "① y ②, sin las franjas en blanco de las horas que el QC descarta."
            )
        else:
            st.info(f"No hay datos suficientes de {nombre_display} para este panel.")

        subseccion(f"Envolvente solar — {nombre_display}: las 4 etapas")
        _fig_env4 = fig_envolvente_comparativo_plotly(_dg4_dfs, _dg4_etq, "GHI", nombre_display)
        if _fig_env4 is not None:
            st.plotly_chart(_fig_env4, use_container_width=True, key=f"envolvente_cmp4_{clave}")
            panel_info(
                "Las 4 envolventes comparten la MISMA escala de color, para que sean "
                "directamente comparables entre sí. Un \"valle oscuro\" que persiste igual "
                "en las 4 versiones es evidencia de obstrucción real del horizonte (no un "
                "artefacto de datos sucios que las etapas previas ya deberían corregir). "
                "Las 4 etapas están a la misma resolución nativa de 5 min (todo el proyecto "
                "trabaja a esa resolución, no a promedios de 1h — ver marco metodológico de "
                "OE2), así que las 4 envolventes deberían verse igual de densas entre sí."
            )
        else:
            st.info(f"No hay datos suficientes de {nombre_display} para la envolvente comparativa.")
    else:
        st.info(f"No hay datos de {nombre_display} para el diagnóstico visual.")


def render_oe3_anexo():
    """Anexo metodológico de OE3 con la implementación en la dirección clásica
    de site-adaptation (serie satelital corregida con la estación terrestre
    como referencia)."""
    st.markdown(
        f"""<div style='background:linear-gradient(135deg,{C_GOLD}cc,{C_TEAL});
        color:white; padding:28px 36px; border-radius:16px; margin-bottom:1.5rem'>
        <div style='font-size:2.25rem; font-weight:600; color:white; margin:0'>📊 OE3 (Anexo) — Desempeño y Re-QC</div>
        <p style='color:#f0f8e8; margin-top:6px; font-size:1rem'>
        Re-aplicación del control de calidad · Comparación pre/post · 802 horas de prueba</p>
        </div>""",
        unsafe_allow_html=True,
    )

    tooltip("""
    **¿Qué se hizo en esta verificación?**

    Esta etapa responde la pregunta que la corrección por sí sola no puede
    contestar: **¿el dato adaptado, además de ser más preciso, sigue siendo
    físicamente aceptable?**

    Para responderla, se re-aplica exactamente el mismo control de calidad
    de la etapa inicial al dato satelital crudo Y al dato adaptado por el
    modelo ganador, **sobre el mismo conjunto de prueba (802 horas)**. Así
    la comparación es con la misma vara.

    Una mejora real requiere:
    - Menor error estadístico (MAE, RMSE, MBE ↓)
    - Mayor porcentaje de datos utilizables físicamente (% usable ↑)
    - Sin nuevos artefactos introducidos por el modelo
    """)

    # Cargar datos
    df_metricas = None
    for ruta in [DIR_TABLAS / "oe3_metricas_test.csv", ROOT / "oe3_metricas_test.csv"]:
        if ruta.exists():
            df_metricas = pd.read_csv(ruta)
            break

    df_mejoras = None
    for ruta in [DIR_TABLAS / "oe3_mejoras_pct_test.csv", ROOT / "oe3_mejoras_pct_test.csv"]:
        if ruta.exists():
            df_mejoras = pd.read_csv(ruta)
            break

    df_usab_oe3 = None
    for ruta in [DIR_TABLAS / "oe3_usabilidad_comparativo.csv", ROOT / "oe3_usabilidad_comparativo.csv"]:
        if ruta.exists():
            df_usab_oe3 = pd.read_csv(ruta)
            break

    # ── KPIs de mejora ────────────────────────────────────────────────────────
    if df_mejoras is not None:
        seccion("Mejoras del dato adaptado vs Solcast crudo")
        row = df_mejoras.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("MAE", f"−{row['mejora_MAE_%']:.1f}%",
                  delta="↓ Mejor", delta_color="normal")
        c2.metric("RMSE", f"−{row['mejora_RMSE_%']:.1f}%",
                  delta="↓ Mejor", delta_color="normal")
        c3.metric("rRMSE", f"−{row['mejora_rRMSE_%']:.1f}%",
                  delta="↓ Mejor", delta_color="normal")
        c4.metric("Pearson R", f"{row['mejora_Pearson_%']:+.2f}%",
                  delta="≈ sin cambio", delta_color="off")
        c5.metric("R²", f"+{row['mejora_R2_%']:.1f}%",
                  delta="↑ Mejor", delta_color="normal")

    # ── Tabla de métricas ─────────────────────────────────────────────────────
    if df_metricas is not None:
        seccion("Métricas de error — Solcast crudo vs Random Forest adaptado")

        if len(df_metricas) >= 2:
            crudo = df_metricas[df_metricas["serie"].str.contains("crudo|Solcast", case=False)].iloc[0]
            adaptado = df_metricas[df_metricas["serie"].str.contains("Forest|adapted", case=False)].iloc[0]

            tabla_metricas = pd.DataFrame({
                "Métrica": ["MAE [W/m²]", "RMSE [W/m²]", "MBE [W/m²]", "rRMSE [%]", "Pearson R", "R²"],
                "Solcast crudo": [
                    f"{crudo['MAE']:.2f}", f"{crudo['RMSE']:.2f}",
                    f"{crudo['MBE']:+.2f}", f"{crudo['rRMSE']:.2f}",
                    f"{crudo['Pearson_R']:.4f}", f"{crudo['R2']:.4f}",
                ],
                "Site-adapted (Random Forest)": [
                    f"{adaptado['MAE']:.2f}", f"{adaptado['RMSE']:.2f}",
                    f"{adaptado['MBE']:+.2f}", f"{adaptado['rRMSE']:.2f}",
                    f"{adaptado['Pearson_R']:.4f}", f"{adaptado['R2']:.4f}",
                ],
                "Mejora": [
                    f"−{100*(crudo['MAE']-adaptado['MAE'])/crudo['MAE']:.1f}%",
                    f"−{100*(crudo['RMSE']-adaptado['RMSE'])/crudo['RMSE']:.1f}%",
                    "−97.1%",
                    f"−{100*(crudo['rRMSE']-adaptado['rRMSE'])/crudo['rRMSE']:.1f}%",
                    f"≈ sin cambio",
                    f"+{100*(adaptado['R2']-crudo['R2'])/crudo['R2']:.1f}%",
                ],
            })

            tabla_html(tabla_metricas)
            boton_descarga_csv(tabla_metricas, "oe3_metricas_comparativas.csv")

    # ── Re-QC ─────────────────────────────────────────────────────────────────
    if df_usab_oe3 is not None:
        seccion("Re-control de calidad — % datos utilizables sobre 802h de prueba")

        df_qc_oe3 = None
        for ruta in [DIR_TABLAS / "oe3_flags_comparativo.csv", ROOT / "oe3_flags_comparativo.csv"]:
            if ruta.exists():
                df_qc_oe3 = pd.read_csv(ruta)
                break

        col_u1, col_u2 = st.columns(2)
        for i, (_, row) in enumerate(df_usab_oe3.iterrows()):
            col = col_u1 if i == 0 else col_u2
            pct = row["pct_usable"]
            color = C_TEAL if "Forest" in str(row["serie"]) else C_BLUE
            col.markdown(
                f"<div style='background:{C_LIGHT}; border-radius:12px; padding:20px; "
                f"border-left:5px solid {color}'>"
                f"<b style='font-size:1rem; color:{color}'>{row['serie']}</b><br>"
                f"<span style='font-size:2.5rem; font-weight:700; color:{color}'>{pct:.2f}%</span>"
                f"<br><small style='color:#666'>{row['n_usable']:,} de {row['n_total']:,} horas utilizables</small>"
                f"</div>", unsafe_allow_html=True
            )

        tooltip("""
        **¿Por qué el dato adaptado es más físicamente consistente?**

        La mejora en usabilidad (91.52% → 92.39%) viene principalmente de la
        **prueba de cielo claro (CSI)**: Solcast sobreestima en condiciones de cielo
        despejado, y esas horas disparan el flag. El Random Forest aprende ese
        patrón y lo corrige, eliminando el 100% de los flags de CSI.

        El resultado es una mejora **simultánea** en precisión estadística
        (RMSE −49%) Y en consistencia física (+0.87 pp de usabilidad) —
        sin ningún costo entre los dos frentes.
        """)

        if df_qc_oe3 is not None:
            subseccion("Detalle de flags por prueba — comparación crudo vs adaptado")
            pruebas_oe3 = {
                "test_negativo": "GHI negativo",
                "test_nocturno": "GHI nocturno",
                "test_limite_fisico": "Límite físico",
                "test_extremely_rare": "Extremadamente raro",
                "test_csi": "Cielo claro (CSI)",
                "test_simetria": "Simetría AM/PM",
            }
            df_qc_show = df_qc_oe3.copy()
            df_qc_show["prueba"] = df_qc_show["prueba"].map(lambda x: pruebas_oe3.get(x, x))
            cols_pivot = ["prueba", "serie", "pct_flag"]
            if all(c in df_qc_show.columns for c in cols_pivot):
                df_pivot = df_qc_show.pivot_table(
                    index="prueba", columns="serie", values="pct_flag", aggfunc="first"
                ).reset_index()
                df_pivot.columns.name = None
                tabla_html(df_pivot, formato={
                    c: "{:.3f}%" for c in df_pivot.columns if c != "prueba"
                })
                boton_descarga_csv(df_pivot, "oe3_flags_comparativo.csv")

    # ── Lámina de evolución del dato ─────────────────────────────────────────
    seccion("Evolución del dato — de crudo a adaptado (todas las etapas)")

    tooltip("""
    **¿Qué muestra esta lámina?**

    Cada barra representa una etapa del pipeline. Las métricas siempre comparan
    Solcast (o el dato adaptado) contra Alvernia como referencia:

    - **Etapa 1 — Crudo:** Solcast vs Alvernia, promedios horarios diurnos, sin QC.
    - **Etapa 2 — Post control de calidad:** 2.671 horas emparejadas con QC (ventana feb–20jul).
    - **Etapa 3 — Post site-adaptation:** modelo ganador vs Alvernia, 802h de prueba.
    - **Etapa 4 — Post control de calidad adaptado:** dato adaptado con re-QC (741h utilizables).

    Tendencia esperada: MAE/RMSE/MBE ↓ y R²/Pearson r ↑.
    """)

    # Tabla + figura de evolución
    _df_evol = None
    for _re in [DIR_TABLAS / "evolucion_metricas_etapas.csv",
                ROOT / "evolucion_metricas_etapas.csv"]:
        if _re.exists():
            _df_evol = pd.read_csv(_re)
            break

    if _df_evol is not None and {"etapa", "MAE", "RMSE"}.issubset(_df_evol.columns):
        _fig_evol = go.Figure()
        _fig_evol.add_trace(go.Bar(name="MAE [W/m²]", x=_df_evol["etapa"], y=_df_evol["MAE"],
                                    marker_color=C_BLUE, opacity=0.85))
        _fig_evol.add_trace(go.Bar(name="RMSE [W/m²]", x=_df_evol["etapa"], y=_df_evol["RMSE"],
                                    marker_color=C_TEAL, opacity=0.85))
        if "MBE" in _df_evol.columns:
            _fig_evol.add_trace(go.Bar(name="MBE [W/m²]", x=_df_evol["etapa"], y=_df_evol["MBE"],
                                        marker_color=C_GOLD, opacity=0.85))
        _fig_evol.update_layout(
            barmode="group", height=420, plot_bgcolor="white", paper_bgcolor="white",
            title="Evolución del dato GHI a través de las 4 etapas del pipeline",
            legend=dict(orientation="h", y=1.05, x=0),
            margin=dict(t=60, b=60, l=40, r=20),
            xaxis=dict(tickangle=-15),
            yaxis=dict(title="W/m²", gridcolor="#f0f0f0"),
        )
        st.plotly_chart(_fig_evol, use_container_width=True, key="evolucion_dato_solar")
    else:
        _fig_evol_ok = mostrar_figura(
            "25_evolucion_dato_solar.png",
            "Evolución del dato GHI a través de las 4 etapas del pipeline."
        )
        if _fig_evol_ok:
            boton_descarga_fig("25_evolucion_dato_solar.png")
        else:
            st.info(
                "💡 Esta lámina se genera con el script de análisis que produce la "
                "figura de evolución del dato a través de las 4 etapas del pipeline."
            )

    if _df_evol is not None:
        subseccion("Tabla resumen de métricas por etapa")
        _cols_ev = {
            "etapa": "Etapa", "n_horas": "N horas",
            "MBE": "MBE [W/m²]", "MAE": "MAE [W/m²]", "RMSE": "RMSE [W/m²]",
            "Pearson_r": "Pearson r", "R2": "R²", "pct_usable": "% usable",
        }
        _df_ev = _df_evol[[c for c in _cols_ev if c in _df_evol.columns]].copy()
        _df_ev = _df_ev.rename(columns=_cols_ev)
        tabla_html(_df_ev, formato={
            "N horas":     "{:,.0f}",
            "MBE [W/m²]":  "{:+.2f}",
            "MAE [W/m²]":  "{:.2f}",
            "RMSE [W/m²]": "{:.2f}",
            "Pearson r":   "{:.3f}",
            "R²":          "{:.3f}",
            "% usable":    "{:.1f}%",
        })
        boton_descarga_csv(_df_ev, "evolucion_metricas_etapas.csv",
                           "⬇ Descargar tabla de evolución (.csv)")

        # KPIs etapa 1 → etapa 4
        try:
            _e1 = _df_evol.iloc[0]
            _e4 = _df_evol.iloc[3]
            subseccion("Mejora total: Etapa 1 (crudo) → Etapa 4 (dato final)")
            _c1, _c2, _c3, _c4, _c5 = st.columns(5)
            _c1.metric("MAE", f"{_e4['MAE']:.1f} W/m²",
                       delta=f"−{100*(_e1['MAE']-_e4['MAE'])/_e1['MAE']:.1f}% vs crudo",
                       delta_color="normal")
            _c2.metric("RMSE", f"{_e4['RMSE']:.1f} W/m²",
                       delta=f"−{100*(_e1['RMSE']-_e4['RMSE'])/_e1['RMSE']:.1f}% vs crudo",
                       delta_color="normal")
            _c3.metric("MBE", f"+{_e4['MBE']:.1f} W/m²",
                       delta=f"−{100*(_e1['MBE']-_e4['MBE'])/_e1['MBE']:.1f}% vs crudo",
                       delta_color="normal")
            _c4.metric("R²", f"{_e4['R2']:.3f}",
                       delta=f"+{100*(_e4['R2']-_e1['R2'])/_e1['R2']:.1f}% vs crudo",
                       delta_color="normal")
            _c5.metric("% usable", f"{_e4['pct_usable']:.1f}%",
                       delta=f"etapa 3→4: +{_e4['pct_usable']-_e1['pct_usable']:.2f} pp",
                       delta_color="normal")
        except Exception:
            pass

    # ── Figuras de evolución de irradiancia ──────────────────────────────────
    seccion("Evolución de la irradiancia GHI — de datos crudos al dato adaptado")

    tooltip(
        "Estas figuras muestran visualmente cómo cambia la señal de GHI en cada etapa:\n\n"
        "- 26A: las 5 fuentes tal cual llegan (datos crudos).\n"
        "- 26B: mismas fuentes pero solo horas diurnas válidas (GHI > 10 W/m²).\n"
        "- 26C: emparejamiento Solcast vs Alvernia en 1H (ventana limpia feb-20jul).\n"
        "- 26D: resultado final — Solcast crudo vs RF adaptado vs Alvernia (test 802h).\n\n"
        "Colores: Solcast=rojo | Alvernia=naranja | HUDN=borgona | "
        "UCC=morado | UDENAR-est=azul marino | RF adaptado=teal.\n\n"
        "Generadas por: generar_fig26_evolucion_irradiancia.py"
    )

    _crudos_26 = cargar_crudos_global()

    def _fig_5fuentes_diario(series_dict, titulo):
        """Serie temporal (promedio diario) con las 5 fuentes superpuestas,
        cada una con su color fijo de COLORES_FUENTES."""
        fig = go.Figure()
        for _n, _s in series_dict.items():
            _sd = _s.dropna().resample("D").mean().dropna()
            if _sd.empty:
                continue
            fig.add_trace(go.Scatter(
                x=_sd.index, y=_sd.values, mode="lines", name=_n,
                line=dict(color=COLORES_FUENTES.get(_n, "#888"), width=1.3),
                opacity=0.85,
            ))
        fig.update_layout(
            title=titulo, height=420, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.05, x=0),
            margin=dict(t=50, b=20, l=60, r=20),
            xaxis=dict(title="Fecha", gridcolor="#f0f0f0"),
            yaxis=dict(title="GHI diario medio [W/m²]", gridcolor="#f0f0f0", rangemode="tozero"),
        )
        return fig

    # 26A — Datos crudos superpuestos
    subseccion("26A — Datos crudos: 5 fuentes superpuestas (sin QC)")
    if _crudos_26:
        st.plotly_chart(_fig_5fuentes_diario(_crudos_26, "GHI diario medio — sin ningún filtro"),
                         use_container_width=True, key="fig26a_crudo")
        panel_info(
            "GHI diario medio de las 5 fuentes superpuestas sin ningún filtro aplicado. "
            "Se aprecia la variación de cobertura temporal entre fuentes y el offset "
            "nocturno de Alvernia a partir del 21-jul-2025."
        )
    else:
        st.info(
            "Estas figuras se generan con el script de análisis de evolución de la irradiancia."
        )

    # 26B — Post-filtro diurno
    subseccion("26B — Post-filtro diurno: solo GHI > 10 W/m² (primer filtro QC)")
    if _crudos_26:
        _crudos_diurnos = {n: s[s > 10] for n, s in _crudos_26.items()}
        st.plotly_chart(_fig_5fuentes_diario(_crudos_diurnos, "GHI diario medio — solo GHI > 10 W/m²"),
                         use_container_width=True, key="fig26b_diurno")
        panel_info(
            "Las mismas 5 fuentes con el primer filtro aplicado: solo registros diurnos "
            "con GHI > 10 W/m². Los huecos indican días sin datos válidos."
        )

    # 26C — Emparejamiento OE2
    subseccion("26C — Emparejamiento para site-adaptation: Solcast 1H vs Alvernia 1H (ventana limpia)")
    _df_pareado = cargar_procesado("solcast_alvernia_pareado_1h.csv")
    if _df_pareado is not None and {"GHI_solcast", "GHI_alvernia"}.issubset(_df_pareado.columns):
        _dfp = _df_pareado.dropna(subset=["GHI_solcast", "GHI_alvernia"]).sort_values("datetime")
        _fig_26c = go.Figure()
        _fig_26c.add_trace(go.Scatter(
            x=_dfp["datetime"], y=_dfp["GHI_solcast"], mode="lines", name="Solcast (1H)",
            line=dict(color=COLORES_FUENTES["Solcast"], width=1.2),
        ))
        _fig_26c.add_trace(go.Scatter(
            x=_dfp["datetime"], y=_dfp["GHI_alvernia"], mode="lines", name="Alvernia (1H)",
            line=dict(color=COLORES_FUENTES["Alvernia"], width=1.2),
            fill="tonexty", fillcolor="rgba(230,57,70,0.12)",
        ))
        _fig_26c.add_vline(x="2025-06-03 05:00:00", line_dash="dash", line_color="#444",
                            annotation_text="Train (70%) | Test (30%)", annotation_position="top")
        _fig_26c.update_layout(
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.05, x=0),
            xaxis=dict(title="Fecha", gridcolor="#f0f0f0"),
            yaxis=dict(title="GHI [W/m²]", gridcolor="#f0f0f0", rangemode="tozero"),
            margin=dict(t=40, b=20, l=60, r=20),
        )
        st.plotly_chart(_fig_26c, use_container_width=True, key="fig26c_emparejamiento")
        panel_info(
            "Solcast y Alvernia promediados a resolución horaria (1H) sobre la ventana limpia "
            "feb-20jul 2025. El área rellena muestra la sobreestimación sistemática de Solcast "
            "(sesgo medio: +34.8 W/m²). La línea vertical separa entrenamiento (70%) de prueba (30%)."
        )
    else:
        _ok_26c = mostrar_figura(
            "26C_emparejamiento_oe2_solcast_alvernia.png",
            "Solcast y Alvernia promediados a resolución horaria (1H) sobre la ventana "
            "limpia feb-20jul 2025."
        )
        if _ok_26c:
            boton_descarga_fig("26C_emparejamiento_oe2_solcast_alvernia.png")

    # 26D — OE3: fuente satelital vs. Random Forest vs. estación terrestre
    subseccion("26D — Verificación final: Solcast crudo vs Random Forest adaptado vs Alvernia (test)")
    _df_pred26d = cargar_procesado("siteadapt_predicciones_test.csv")
    if _df_pred26d is not None and {"GHI_alvernia", "Baseline (Solcast sin corregir)", "Random Forest"}.issubset(_df_pred26d.columns):
        _dfd = _df_pred26d.sort_values("datetime")
        _fig_26d = go.Figure()
        _fig_26d.add_trace(go.Scatter(
            x=_dfd["datetime"], y=_dfd["Baseline (Solcast sin corregir)"], mode="lines",
            name="Solcast crudo", line=dict(color=COLORES_FUENTES["Solcast"], width=1.3, dash="dot"),
        ))
        _fig_26d.add_trace(go.Scatter(
            x=_dfd["datetime"], y=_dfd["GHI_alvernia"], mode="lines",
            name="Alvernia (referencia)", line=dict(color=COLORES_FUENTES["Alvernia"], width=1.6),
        ))
        _fig_26d.add_trace(go.Scatter(
            x=_dfd["datetime"], y=_dfd["Random Forest"], mode="lines",
            name="Random Forest (adaptado)", line=dict(color=C_TEAL, width=1.6),
        ))
        _fig_26d.update_layout(
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.05, x=0),
            xaxis=dict(title="Fecha (802h de prueba)", gridcolor="#f0f0f0"),
            yaxis=dict(title="GHI [W/m²]", gridcolor="#f0f0f0", rangemode="tozero"),
            margin=dict(t=40, b=20, l=60, r=20),
        )
        st.plotly_chart(_fig_26d, use_container_width=True, key="fig26d_oe3_comparacion")
        panel_info(
            "Las 3 series sobre las 802 horas de prueba (jun-20jul 2025). "
            "El Random Forest adaptado (teal) sigue de cerca a Alvernia (naranja), "
            "mientras Solcast crudo (negro punteado) sobreestima consistentemente."
        )
    else:
        _ok_26d = mostrar_figura(
            "26D_oe3_solcast_vs_rf_vs_alvernia.png",
            "Las 3 series sobre las 802 horas de prueba (jun-20jul 2025)."
        )
        if _ok_26d:
            boton_descarga_fig("26D_oe3_solcast_vs_rf_vs_alvernia.png")

    panel_info(
        "La secuencia 26A → 26B → 26C → 26D muestra visualmente la trayectoria completa "
        "del dato: de 5 fuentes crudas con sesgo y ruido, al dato satelital corregido "
        "que converge con la medicion terrestre de referencia. "
        "El salto mas notable ocurre entre 26C y 26D: el Random Forest elimina "
        "el sesgo de +34.8 W/m² y reduce el MAE de 44.2 a 18.7 W/m² (−57.7%)."
    )

    # ── Figuras de desempeño ──────────────────────────────────────────────────
    seccion("Visualizaciones de desempeño")

    _df_pred15 = cargar_procesado("siteadapt_predicciones_test.csv")
    if _df_pred15 is not None and {"GHI_alvernia", "Baseline (Solcast sin corregir)", "Random Forest"}.issubset(_df_pred15.columns):
        _dfp15 = _df_pred15.dropna(subset=["GHI_alvernia", "Baseline (Solcast sin corregir)", "Random Forest"])
        _lim15 = float(max(
            _dfp15["GHI_alvernia"].max(),
            _dfp15["Baseline (Solcast sin corregir)"].max(),
            _dfp15["Random Forest"].max(),
        )) * 1.05

        _fig15 = make_subplots(rows=1, cols=2, subplot_titles=(
            "Solcast crudo vs Alvernia", "Random Forest vs Alvernia"))
        _fig15.add_trace(go.Scatter(
            x=_dfp15["GHI_alvernia"], y=_dfp15["Baseline (Solcast sin corregir)"],
            mode="markers", marker=dict(color=COLORES_FUENTES["Solcast"], size=4, opacity=0.5),
            name="Solcast crudo", showlegend=False,
        ), row=1, col=1)
        _fig15.add_trace(go.Scatter(
            x=[0, _lim15], y=[0, _lim15], mode="lines",
            line=dict(color="black", dash="dash", width=1), name="1:1", showlegend=False,
        ), row=1, col=1)
        _fig15.add_trace(go.Scatter(
            x=_dfp15["GHI_alvernia"], y=_dfp15["Random Forest"],
            mode="markers", marker=dict(color=C_TEAL, size=4, opacity=0.5),
            name="Random Forest", showlegend=False,
        ), row=1, col=2)
        _fig15.add_trace(go.Scatter(
            x=[0, _lim15], y=[0, _lim15], mode="lines",
            line=dict(color="black", dash="dash", width=1), name="1:1", showlegend=False,
        ), row=1, col=2)
        _fig15.update_xaxes(title_text="GHI Alvernia [W/m²]", gridcolor="#f0f0f0")
        _fig15.update_yaxes(title_text="GHI predicho [W/m²]", gridcolor="#f0f0f0", col=1)
        _fig15.update_layout(
            height=440, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40, b=20, l=60, r=20),
        )
        st.plotly_chart(_fig15, use_container_width=True, key="fig15_dispersion")
        panel_info(
            "Dispersión GHI: Solcast crudo vs Alvernia (izq.) y Random Forest vs Alvernia (der.). "
            "La nube de puntos de la derecha se ajusta visiblemente mejor a la diagonal 1:1."
        )

        _fig16 = go.Figure()
        _dfp16 = _df_pred15.sort_values("datetime")
        _fig16.add_trace(go.Scatter(
            x=_dfp16["datetime"], y=_dfp16["GHI_alvernia"], mode="lines",
            name="Alvernia (referencia)", line=dict(color=COLORES_FUENTES["Alvernia"], width=1.6),
        ))
        _fig16.add_trace(go.Scatter(
            x=_dfp16["datetime"], y=_dfp16["Baseline (Solcast sin corregir)"], mode="lines",
            name="Solcast crudo", line=dict(color=COLORES_FUENTES["Solcast"], width=1.3, dash="dot"),
        ))
        _fig16.add_trace(go.Scatter(
            x=_dfp16["datetime"], y=_dfp16["Random Forest"], mode="lines",
            name="Random Forest (adaptado)", line=dict(color=C_TEAL, width=1.6),
        ))
        _fig16.update_layout(
            height=400, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.05, x=0),
            xaxis=dict(title="Fecha (802h de prueba)", gridcolor="#f0f0f0"),
            yaxis=dict(title="GHI [W/m²]", gridcolor="#f0f0f0", rangemode="tozero"),
            margin=dict(t=40, b=20, l=60, r=20),
        )
        st.plotly_chart(_fig16, use_container_width=True, key="fig16_serie_antes_despues")
        panel_info(
            "Serie temporal comparativa sobre el periodo de prueba: "
            "Alvernia (referencia), Solcast crudo y Random Forest adaptado."
        )
    else:
        mostrar_figura("15_dispersion_antes_despues.png",
                       "Dispersión GHI: Solcast crudo vs Alvernia (izq.) y Random Forest vs Alvernia (der.).")
        boton_descarga_fig("15_dispersion_antes_despues.png")

        mostrar_figura("16_serie_temporal_antes_despues.png",
                       "Serie temporal comparativa sobre el periodo de prueba.")
        boton_descarga_fig("16_serie_temporal_antes_despues.png")

    st.markdown("<br>", unsafe_allow_html=True)

    # Conclusión
    st.markdown(
        f"""<div style='background:linear-gradient(135deg,{C_TEAL}22,{C_BLUE}11);
        border:2px solid {C_TEAL}; border-radius:14px; padding:24px 28px; margin-top:1rem'>
        <div style='font-size:1.25rem; font-weight:600; color:{C_TEAL}; margin-top:0'>✅ Conclusión (Anexo) — caso específico Solcast/Alvernia</div>
        <p style='font-size:1.02rem; color:#222; margin-bottom:0'>
        Para este par específico (dirección clásica: Solcast corregido con Alvernia como
        referencia), el modelo ganador mejora simultáneamente precisión estadística
        y consistencia física respecto al Solcast crudo:
        <b>MAE −57.7% · RMSE −49.0% · R² +24.5% · usabilidad +0.87 pp · flags CSI 0.998% → 0.00%</b>.
        Confirma que el pipeline completo produce, en este caso, un dato solar
        corregido, trazable y físicamente defendible.
        </p></div>""",
        unsafe_allow_html=True,
    )


def render_inicio_resumen():
    """Primera pestaña de Inicio: resumen del proyecto, mapa, fuentes, KPIs y
    datos crudos. La segunda pestaña contiene el marco metodológico de métricas."""
    st.markdown(
        f"""<div style='background:linear-gradient(135deg,{C_TEAL},{C_BLUE});
        color:white; padding:36px 40px; border-radius:16px; margin-bottom:2rem'>
        <div style='color:white; margin:0; font-size:2.2rem; font-weight:600'>☀️ Control de Calidad y Site-Adaptation</div>
        <p style='color:#d0eaf4; font-size:1.1rem; margin-top:8px'>
        Análisis del desempeño de técnicas de QC y Site-Adaptation en datos de Recurso Solar<br>
        <strong>San Juan de Pasto, Nariño — 2025</strong></p></div>""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div style='background:{C_LIGHT}; border-radius:12px; padding:20px;
        border-top:4px solid {C_TEAL}'>
        <div style='font-size:1.25rem; font-weight:600; color:{C_TEAL}; margin-top:0'>🔬 Control de Calidad</div>
        <small>Protocolo de control de calidad (pruebas físicas y de experto, estándar BSRN)
        aplicado a 2 casos independientes, mismo sitio: Solcast+Udenar (2023, inversor) y
        Solcast+{NOMBRE_UDENAR_EST} (estación meteorológica real).</small>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div style='background:{C_LIGHT}; border-radius:12px; padding:20px;
        border-top:4px solid {C_BLUE}'>
        <div style='font-size:1.25rem; font-weight:600; color:{C_BLUE}; margin-top:0'>📐 Site-Adaptation</div>
        <small>9 modelos de corrección de sesgo comparados. Se selecciona automáticamente el de
        mejor desempeño entre los que producen predicciones físicamente válidas.</small></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div style='background:{C_LIGHT}; border-radius:12px; padding:20px;
        border-top:4px solid {C_GOLD}'>
        <div style='font-size:1.25rem; font-weight:600; color:{C_GOLD}; margin-top:0'>📊 Desempeño</div>
        <small>Re-aplicación del control de calidad al dato adaptado, verificando que la
        corrección mejore simultáneamente la precisión estadística y la consistencia física.</small>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Contexto
    with st.expander("📖  Contexto del proyecto", expanded=True):
        st.markdown("""
        El recurso solar de un sitio puede estimarse a partir de **productos satelitales** (Solcast),
        que ofrecen cobertura espacial y temporal continua, o de **estaciones terrestres**, que miden
        directamente pero suelen tener huecos y ruido instrumental.

        Este proyecto construye un flujo reproducible, aplicado a **2 casos independientes**
        (mismo sitio, distinto año/estación terrestre) que:
        1. **Controla la calidad** de cada par Solcast + estación terrestre con un protocolo
           único y trazable.
        2. **Corrige y completa** el dato de la estación terrestre usando Solcast —la única
           fuente sin huecos temporales— como referencia: una *reconstrucción inversa* de
           *site-adaptation*.
        3. **Verifica** que esa corrección no introduzca artefactos no físicos, re-aplicando el
           mismo control de calidad al dato reconstruido.

        El software se desarrolló como un **paquete Python instalable** (`qcsolar`) con scripts
        de verificación reproducibles. Todos los resultados mostrados aquí provienen de corridas
        reales sobre los datos del proyecto.
        """)

    # Mapa + fuentes
    col_mapa, col_fuentes = st.columns([1.2, 1])
    with col_mapa:
        subseccion("Ubicación — Universidad de Nariño (Udenar), Pasto")
        st.map(pd.DataFrame({
            "lat": [1.2332026137776049],
            "lon": [-77.29287682225166],
        }), zoom=13, use_container_width=True)
        panel_info("Altitud: 2.451 m s.n.m. · Latitud: 1.2332° N · Longitud: 77.2929° O — "
                    "el pixel de Solcast y la estación terrestre corresponden al mismo punto.")

    with col_fuentes:
        subseccion("Fuentes de datos")
        st.markdown(f"""
        | Fuente | Tipo | Resolución | Período |
        |--------|------|------------|---------|
        | **Solcast** | Satelital | 5 min | 2023 |
        | **Udenar** | Estación terrestre (inversor) | 5 min | jul-dic 2023 |
        | **Solcast 2025** | Satelital | 5 min | 2025 |
        | **{NOMBRE_UDENAR_EST}** | Estación meteorológica real | 1 min | 2025 |
        """)
        st.markdown("<br>", unsafe_allow_html=True)

        # KPIs globales
        df_u = cargar_csv("usabilidad_comparativo.csv", ROOT / "outputs" / "tablas")
        if df_u is None:
            df_u = pd.read_csv(ROOT / "usabilidad_comparativo.csv") if (ROOT / "usabilidad_comparativo.csv").exists() else None

        if df_u is not None:
            subseccion("% Datos utilizables (regla base)")
            for _, row in df_u.iterrows():
                fuente = FUENTES_LABEL.get(row["fuente"], row["fuente"])
                pct = row["pct_usable_base"]
                color = C_TEAL if pct > 70 else C_GOLD if pct > 50 else C_RED
                barra_pct = min(int(pct), 100)
                st.markdown(
                    f"<div style='margin:4px 0'><small>{fuente}</small><br>"
                    f"<div style='background:#e0edf2; border-radius:6px; height:18px; width:100%'>"
                    f"<div style='background:{color}; width:{barra_pct}%; height:100%; border-radius:6px; "
                    f"display:flex; align-items:center; padding-left:8px'>"
                    f"<small style='color:white; font-weight:700'>{pct:.1f}%</small></div></div></div>",
                    unsafe_allow_html=True,
                )

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN: DATOS CRUDOS — visualización inicial
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    seccion("Datos crudos — Irradiancia GHI sin ningún procesamiento")

    tooltip(
        "Esta sección muestra los datos tal como llegan de cada fuente, sin ningún filtro "
        "ni control de calidad aplicado. Es el punto de partida del pipeline.\n\n"
        "Cada fuente tiene su propia resolución temporal, período de cobertura y unidades. "
        "Los datos se promedian a resolución diaria únicamente para la visualización — "
        "el pipeline opera sobre los datos originales sin este promedio.\n\n"
        "Los colores son consistentes en esta sección — satelital=azul oscuro, "
        "terrestre=verde oscuro (cada caso con su propio tono):\n"
        "Solcast=azul oscuro | Udenar (terrestre)=verde oscuro | "
        "Solcast 2025=azul oscuro | "
        f"{NOMBRE_UDENAR_EST}=verde oscuro"
    )

    # Paleta de colores por fuente para esta sección: satelital en tonos de
    # azul oscuro, terrestre en tonos de verde oscuro; cada
    # caso (2023/2025) con su propio tono dentro de esa familia para poder
    # distinguirlos.
    COLORES_FUENTES = {
        "Solcast":            "#154360",   # azul oscuro (caso 2023)
        "Udenar (terrestre)": "#145A32",   # verde oscuro (caso 2023)
        "Solcast 2025":       "#1F618D",   # azul oscuro, otro tono (caso 2025)
        NOMBRE_UDENAR_EST:    "#0E6251",   # verde oscuro, otro tono (caso 2025)
    }

    # ── Cargar y preparar datos crudos ───────────────────────────────────────
    # Reutiliza cargar_crudos_global(), que ya cubre los 2 casos de estudio
    # (fuente satelital + estación terrestre, 2023 y 2025).
    _crudos = cargar_crudos_global()

    if not _crudos:
        st.warning(
            "No se encontraron los archivos CSV crudos en la raíz del proyecto. "
            "Verifica que los CSV originales estén en la carpeta raíz."
        )
    else:
        import plotly.graph_objects as go

        # ── Resumen estadístico de datos crudos ──────────────────────────────
        subseccion("Resumen estadístico — datos crudos (GHI diario diurno)")

        _filas_stats = []
        for _nombre, _serie in _crudos.items():
            _s = _serie.dropna()
            # N registros totales y N días únicos
            _n_dias = _s.index.normalize().nunique() if len(_s) > 0 else 0
            _filas_stats.append({
                "Fuente": _nombre,
                "Período": f"{_s.index.min().date()} → {_s.index.max().date()}"
                           if len(_s) > 0 else "—",
                "N registros": f"{len(_s):,}",
                "N días": f"{_n_dias:,}",
                "Media [W/m²]": f"{_s.mean():.1f}",
                "Máx [W/m²]":  f"{_s.max():.1f}",
                "Mín [W/m²]":  f"{_s.min():.1f}",
                "Std [W/m²]":  f"{_s.std():.1f}",
            })
        _df_stats = pd.DataFrame(_filas_stats)
        tabla_html(_df_stats)

        # ── Gráfica por caso (satelital + estación terrestre superpuestos) ────
        subseccion("GHI crudo — Solcast vs. estación terrestre, por caso (resolución original)")

        tooltip(
            "Cada panel superpone Solcast y su estación terrestre pareja para UN caso "
            "(2023 o 2025), sin ningún filtro. Se incluyen TODOS los datos disponibles; "
            "los huecos en la línea indican días sin registros en esa fuente."
        )

        _casos_crudos = [
            ("Caso 2023", [("Solcast", _crudos.get("Solcast")),
                            ("Udenar (terrestre)", _crudos.get("Udenar (terrestre)"))]),
            ("Caso 2025", [("Solcast 2025", _crudos.get("Solcast 2025")),
                            (NOMBRE_UDENAR_EST, _crudos.get(NOMBRE_UDENAR_EST))]),
        ]
        _BOTONES_RANGO = [
            dict(count=7,  label="1S", step="day",   stepmode="backward"),
            dict(count=1,  label="1M", step="month", stepmode="backward"),
            dict(count=3,  label="3M", step="month", stepmode="backward"),
            dict(count=6,  label="6M", step="month", stepmode="backward"),
            dict(step="all", label="Todo"),
        ]

        for _titulo_caso, _fuentes_caso in _casos_crudos:
            _fig_caso = go.Figure()
            for _nombre_f, _serie_f in _fuentes_caso:
                if _serie_f is None:
                    continue
                _s = _serie_f.dropna()
                _fig_caso.add_trace(go.Scattergl(
                    x=_s.index, y=_s.values,
                    mode="lines",
                    line=dict(color=COLORES_FUENTES[_nombre_f], width=1.1),
                    opacity=0.85,
                    name=_nombre_f,
                    hovertemplate=f"<b>{_nombre_f}</b><br>Fecha: %{{x|%d %b %Y %H:%M}}"
                                  "<br>GHI: %{y:.1f} W/m²<extra></extra>",
                ))
            # Leyenda vertical a la derecha del panel para que no compita por
            # espacio con los botones de rango (1S/1M/.../Todo), ubicados arriba.
            _fig_caso.update_layout(
                title=dict(text=_titulo_caso, font=dict(size=16, color=C_TEAL)),
                height=420,
                margin=dict(t=50, b=20, l=60, r=150),
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(
                    orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                    font=dict(size=12), bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1,
                ),
                xaxis=dict(
                    showgrid=False,
                    tickformat="%b %Y",
                    tickfont=dict(size=10),
                    rangeselector=dict(
                        buttons=_BOTONES_RANGO,
                        bgcolor="#f0f6f9",
                        activecolor="#1F4E5F",
                        font=dict(size=10),
                    ),
                    rangeslider=dict(visible=True, thickness=0.07),
                ),
                yaxis=dict(
                    gridcolor="#f0f0f0",
                    title=dict(text="GHI [W/m²]", font=dict(size=11)),
                    tickfont=dict(size=10),
                ),
            )
            st.plotly_chart(_fig_caso, use_container_width=True,
                            key=f"crudo_caso_{_titulo_caso}")


def render_inicio_marco_metricas():
    """Renderiza el marco metodológico de las métricas usadas en OE1, OE2 y
    OE3: fórmula, interpretación y ejemplo con datos reales por métrica."""
    seccion("Marco metodológico — Métricas de evaluación")

    st.markdown("""
    El proyecto usa **3 familias de métricas**, cada una respondiendo una
    pregunta distinta, siempre calculadas sobre el mismo conjunto de datos
    (nunca mezclando entrenamiento con prueba):

    - **Métricas de error de regresión** (MAE, RMSE, MBE, rMBE, rRMSE,
      Pearson R, R²) — comparan una serie predicha/corregida contra una
      serie de referencia medida. Se usan para entrenar y elegir el modelo
      ganador en la etapa de site-adaptation, y para evaluar la mejora en la
      etapa de verificación de desempeño.
    - **Usabilidad QC** (% usable) — de la etapa de control de calidad, mide
      qué fracción de los datos pasa TODAS las pruebas de control de calidad
      evaluables.
    - **Composición del origen del dato** (medido_ok / medido_flag /
      sin_dato) — específica de la reconstrucción en la etapa de
      site-adaptation, mide cuánto del dato final es medición real vs.
      corrección vs. relleno de huecos.

    Todas siguen exactamente la misma implementación centralizada del
    proyecto — los mismos números que ves en cada pestaña.
    """)

    subseccion("Familia 1 — Métricas de error de regresión")

    with st.expander("📏 MAE — Error Absoluto Medio", expanded=True):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Fórmula:**")
            st.latex(r"MAE = \frac{1}{n}\sum_{i=1}^{n} |y_{pred,i} - y_{real,i}|")
            st.markdown("""
            Promedio del valor absoluto del error, en las **mismas unidades**
            que el dato (W/m²). Fácil de interpretar: "en promedio, el modelo
            se equivoca por X W/m²". No distingue si el error es positivo o
            negativo, y todos los errores pesan igual (no penaliza más los
            errores grandes que los pequeños).
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, prueba):**")
            st.info("""
            Baseline (Solcast sin corregir): **MAE = 64.20 W/m²**
            Udenar adaptado (modelo ganador, Extra Trees): **MAE = 47.10 W/m²**

            En promedio, el modelo ganador se equivoca por 47.1 W/m² en vez
            de 64.2 W/m² — una reducción del 26.6%.
            """)

    with st.expander("📏 RMSE — Raíz del Error Cuadrático Medio"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Fórmula:**")
            st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_{pred,i} - y_{real,i})^2}")
            st.markdown("""
            Como MAE, pero eleva el error al cuadrado antes de promediar y
            luego saca raíz — esto **penaliza más los errores grandes**
            (outliers) que los pequeños. Es la métrica que este proyecto usa
            como **criterio de selección del modelo ganador** en la etapa de
            site-adaptation (entre los modelos físicamente válidos, gana el
            de menor RMSE).
            """)
        with col2:
            st.markdown("**Ejemplo real — criterio de selección:**")
            st.info("""
            Udenar adaptado (modelo ganador, regresión lineal simple):
            **RMSE = 98.60 W/m²** — no es el menor RMSE absoluto de los
            candidatos (Gradient Boosting logra 95.82 W/m²), pero sí el
            menor **entre los modelos que pasan el filtro de validez física
            riguroso** (sobre todo el período de aplicación).

            Gradient Boosting, pese a tener el mejor RMSE puro, queda
            descalificado: **12.6% de sus predicciones sobre el año
            completo violan algún límite físico BSRN** — muy por encima del
            pequeño margen de tolerancia reservado solo para casos
            triviales de crepúsculo. RMSE decide **entre** los modelos ya
            filtrados por validez física, no antes del filtro.
            """)

    with st.expander("📏 MBE — Sesgo Medio (Mean Bias Error)"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Fórmula:**")
            st.latex(r"MBE = \frac{1}{n}\sum_{i=1}^{n} (y_{pred,i} - y_{real,i})")
            st.markdown("""
            A diferencia de MAE/RMSE, **conserva el signo** del error: mide
            si el modelo sobreestima (MBE > 0) o subestima (MBE < 0) *en
            promedio*. Un MBE cercano a 0 **no significa error cero** — solo
            que los errores positivos y negativos se cancelan entre sí. Por
            eso siempre se reporta junto con MAE/RMSE, nunca solo.
            """)
        with col2:
            st.markdown("**Ejemplo real — el sesgo que motivó el site-adaptation:**")
            st.error("""
            Solcast crudo vs. Udenar: **MBE = +42.11 W/m²** (rMBE +28.92%)

            Solcast sobreestima sistemáticamente el GHI de Udenar — este
            sesgo constante es exactamente lo que el site-adaptation
            corrige. Tras la corrección (Extra Trees), el MBE cae a **-5.09
            W/m²** (rMBE -3.49%): mucho menor en magnitud, aunque cambia de
            signo — de sobreestimar a subestimar levemente.
            """)

    with st.expander("📏 rMBE y rRMSE — versiones relativas (%)"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Fórmula:**")
            st.latex(r"rMBE = 100 \times \frac{MBE}{\overline{y_{real}}} \qquad "
                      r"rRMSE = 100 \times \frac{RMSE}{\overline{y_{real}}}")
            st.markdown("""
            MAE, RMSE y MBE están en W/m² — su magnitud depende del nivel de
            irradiancia promedio de cada estación. rMBE y rRMSE dividen por
            la **media de la referencia**, expresando el error como
            porcentaje. Esto permite comparar el desempeño del modelo entre
            estaciones o proyectos con niveles de GHI muy distintos de
            forma justa.
            """)
        with col2:
            st.markdown("**Ejemplo real:**")
            st.info("""
            Udenar adaptado (prueba, regresión lineal simple): **RMSE =
            98.60 W/m² → rRMSE ≈ 67.3%**. Baseline: **RMSE = 123.64 W/m² →
            rRMSE ≈ 84.4%**.

            Se usa junto al RMSE absoluto en las tablas comparativas de
            site-adaptation y verificación final precisamente para poder
            comparar estaciones o proyectos con niveles de GHI distintos
            sin que el nivel medio distorsione la lectura.
            """)

    with st.expander("📏 Pearson R — Coeficiente de correlación"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Fórmula:**")
            st.latex(r"r = \frac{\mathrm{cov}(y_{real}, y_{pred})}"
                      r"{\sigma_{y_{real}} \cdot \sigma_{y_{pred}}}")
            st.markdown("""
            Mide qué tan bien el modelo sigue la **forma** de la señal real
            (sube cuando sube, baja cuando baja) — **sin importar el sesgo
            ni la escala**. Un modelo puede tener Pearson R alto y aun así
            estar sistemáticamente desviado: sigue el patrón, pero no el
            nivel correcto.
            """)
        with col2:
            st.markdown("**Ejemplo real — por qué Pearson R solo no basta:**")
            st.error("""
            Solcast crudo vs. Udenar: **Pearson R ≈ 0.897** (alto)
            pero **MBE = +42.11 W/m²** (sesgo grande)

            Solcast sigue muy bien la forma de la señal (nubes, ciclo
            diario), pero está sistemáticamente desplazado hacia arriba.
            Tras la corrección (Extra Trees), Pearson R apenas cambia
            (**0.897 → 0.895**, prácticamente igual) porque ambas series ya
            seguían bien la forma — la corrección real está en el sesgo, no
            en la forma. Por eso Pearson R nunca se usa solo — siempre
            junto a MBE/R².
            """)

    with st.expander("📏 R² — Coeficiente de determinación"):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Fórmula:**")
            st.latex(r"R^2 = 1 - \frac{\sum (y_{real,i} - y_{pred,i})^2}"
                      r"{\sum (y_{real,i} - \overline{y_{real}})^2}")
            st.markdown("""
            Fracción de la varianza de la referencia que el modelo logra
            explicar. A diferencia de Pearson R (que solo mide forma), R²
            **sí penaliza el sesgo** — un modelo desplazado sistemáticamente
            puede tener Pearson R alto pero R² bajo. Es más exigente y es la
            métrica usada en los KPIs de "mejora" de la verificación final.
            """)
        with col2:
            st.markdown("**Ejemplo real:**")
            st.info("""
            Udenar: Baseline **R² = 0.690** → Udenar adaptado **R² = 0.800**

            El salto de R² (+11.0 puntos) contrasta con Pearson R, que
            **prácticamente no cambia** (0.897 → 0.895, incluso baja
            levemente) — porque R² sí penaliza el sesgo sistemático que el
            modelo corrige, y Pearson R (que solo mide forma) no.
            """)

    st.markdown("---")
    subseccion("Familia 2 — Usabilidad QC (% usable)")

    with st.expander("✅ % Usable — regla de control de calidad", expanded=True):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Regla:**")
            st.latex(r"\%\,usable = 100 \times \frac{n_{OK}}{n_{total}}")
            st.markdown("""
            Un dato es **OK** si NINGUNA prueba evaluable de QC lo marca
            `FLAG`; si al menos una lo marca, es **NO_USABLE**. Las pruebas
            `NO_EVALUABLE` (ej: pruebas diurnas de noche) no cuentan ni a
            favor ni en contra. Es la métrica central de la etapa de control
            de calidad y del re-control de calidad en la verificación final.
            """)
        with col2:
            st.markdown("**Ejemplo real (verificación final, Udenar):**")
            st.info("""
            Crudo: **99.85%** usable → Corregido (total): **100.0%** usable

            En Udenar el crudo ya partía con un %usable muy alto, así que
            la reconstrucción tenía poco margen para mejorar ese número —
            aun así sube levemente (+0.15 pp), y el error estadístico
            (RMSE, MAE, R²) también mejora claramente frente al baseline.
            Son dos preguntas distintas, y no siempre se mueven juntas —
            en este caso, ambas confirman la misma conclusión.
            """)

    st.markdown("---")
    subseccion("Familia 3 — Composición del origen del dato (específica de la reconstrucción)")

    with st.expander("🧩 medido_ok / medido_flag / sin_dato", expanded=True):
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.markdown("**Clasificación por hora, en el período de aplicación:**")
            st.markdown("""
            - **medido_ok:** la estación tenía dato con usabilidad OK esa
              hora — el valor final es la medición real (variante selectiva)
              o el modelo, que en la práctica casi lo reproduce (variante
              total).
            - **medido_flag:** la estación tenía dato esa hora, pero no pasó
              el control de calidad — el valor final es la predicción del
              modelo (corrige el dato malo).
            - **sin_dato:** la estación no tenía ningún registro esa hora
              (hueco real) — el valor final es 100% predicción del modelo
              (rellena el hueco).

            Esta clasificación es la que distingue la variante *selectiva*
            (conserva medido_ok tal cual) de la *total* (siempre usa la
            predicción del modelo, igual que el paper de referencia).
            """)
        with col2:
            st.markdown("**Ejemplo real (Udenar, período completo):**")
            st.info("""
            **50.19%** medido_ok · **0.16%** medido_flag (corregido)
            **49.65%** sin_dato (rellenado)

            Casi la mitad del período de aplicación de Udenar no tenía
            ningún dato propio (huecos reales del inversor, no solo datos
            flageados) — la reconstrucción con Solcast como referencia es
            lo que permite rellenarlo. El % ya medido y válido (medido_ok)
            es alto y casi no necesitó corrección puntual (medido_flag muy
            bajo), lo que explica por qué el %usable global ya partía tan
            alto y tenía poco margen de mejora.
            """)

    panel_info(
        "Estas 3 familias de métricas se combinan en las tablas de "
        "'Comparación de variantes' (site-adaptation) y 'Re-control de "
        "calidad' (verificación final): el error de regresión decide QUÉ "
        "modelo gana, la usabilidad QC mide si la reconstrucción es "
        "físicamente aceptable, y la composición del origen del dato "
        "muestra CUÁNTO del resultado final es medición real vs. corrección "
        "vs. relleno."
    )


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAMAS DE FLUJO (Anexo de cada objetivo específico)
# Sintetizan, en cajas y flechas, el pipeline REAL de código de cada OE
# (mismos módulos/funciones ya usados en las pestañas de Marco metodológico
# de arriba). Usan Mermaid (ver render_mermaid) en vez de una imagen fija
# para que sean fáciles de mantener si el pipeline cambia: basta editar el
# texto de las cajas, no volver a dibujar nada.
# ─────────────────────────────────────────────────────────────────────────────
_MERMAID_ESTILOS = """
    classDef entrada fill:#f0f6f9,stroke:#1F4E5F,stroke-width:1.5px,color:#1a1a1a;
    classDef proceso fill:#2b6f8f,stroke:#1F4E5F,stroke-width:1.5px,color:#ffffff;
    classDef decision fill:#e8a838,stroke:#7a5a10,stroke-width:1.5px,color:#1a1a1a;
    classDef variante fill:#F4A261,stroke:#7a5a10,stroke-width:1.5px,color:#1a1a1a;
    classDef salida fill:#1F4E5F,stroke:#1F4E5F,stroke-width:2px,color:#ffffff;
    classDef marcado fill:#f8d7da,stroke:#d1495b,stroke-width:1.5px,color:#7a1020;
    classDef descartado fill:#d1495b,stroke:#7a1020,stroke-width:2px,color:#ffffff;
"""

_MERMAID_OE1 = """flowchart TD
    A["📥 Dato crudo original<br/>(un timestamp) + geometría solar<br/>zenith · elevación · azimut · GHI cielo despejado (pvlib)"]

    subgraph FASE_FIS["🔎 QC FÍSICO — qc/physical.py<br/>(las 5 pruebas evalúan el MISMO dato crudo, en paralelo, no en cadena)"]
        Q1{"¿GHI/DNI/DHI menor al<br/>mínimo físico (−4 W/m²)?<br/>test_negativo()"}
        F1["🚩 flag_neg = FLAG"]
        Q2{"¿Sol bajo el horizonte<br/>y GHI > umbral nocturno?<br/>test_nocturno()"}
        F2["🚩 flag_noche = FLAG"]
        Q3{"¿Irradiancia supera el límite<br/>físico posible BSRN (PPL)?<br/>test_limite_fisico()"}
        F3["🚩 flag_ppl = FLAG"]
        Q4{"¿Irradiancia supera el límite<br/>extremadamente raro BSRN (ERL)?<br/>test_extremely_rare()"}
        F4["🚩 flag_erl = FLAG"]
        Q5{"¿Sensor caído?<br/>TEMP=0 y RH=0 a la vez<br/>(solo estaciones terrestres)<br/>test_sensor_caido()"}
        F5["🚩 flag_sensor_caido = FLAG"]
    end

    subgraph FASE_EXP["🧪 QC EXPERTO — qc/expert.py<br/>(idem: 6 pruebas en paralelo sobre el mismo dato crudo)"]
        Q6{"¿Falla el cierre radiativo<br/>GHI ≈ DNI·cosθ + DHI?<br/>(solo Solcast) · test_cierre()"}
        F6["🚩 flag_cierre = FLAG"]
        Q7{"¿Violan Kt, Kn o K<br/>sus límites físicos?<br/>(solo Solcast) · test_ktests()"}
        F7["🚩 flag_ktests = FLAG"]
        Q8{"¿CSI = GHI/GHI_cielo_despejado<br/>supera el máximo?<br/>test_csi()"}
        F8["🚩 flag_csi = FLAG"]
        Q9{"¿Salto entre 2 muestras<br/>consecutivas > máximo?<br/>test_rate_of_change()"}
        F9["🚩 flag_roc = FLAG"]
        Q10{"¿El mismo valor se repite<br/>≥ N muestras seguidas?<br/>test_persistencia()"}
        F10["🚩 flag_persistencia = FLAG"]
        Q11{"¿Asimetría AM/PM del día<br/>supera la tolerancia?<br/>test_simetria_ampm()"}
        F11["📋 flag_simetria = FLAG<br/>(diagnóstico — NO entra<br/>a la decisión final)"]
    end

    Q12{"¿ALGUNA de las 10 pruebas<br/>evaluables (todas menos<br/>simetría) marcó FLAG<br/>para este dato?<br/>calcular_usabilidad()"}
    Q13{"¿Dato OK aislado, rodeado<br/>de tramos NO_USABLE a<br/>menos de 60 min?<br/>aplicar_regla_islas() [opcional]"}
    R1[("🚫 NO_USABLE — se descarta")]
    R2[("✅ OK — dataset que PASÓ el control de calidad")]

    A --> Q1
    A --> Q2
    A --> Q3
    A --> Q4
    A --> Q5
    A --> Q6
    A --> Q7
    A --> Q8
    A --> Q9
    A --> Q10
    A --> Q11

    Q1 -->|Sí, falla| F1 --> Q12
    Q1 -.->|No, pasa OK, sin consecuencia| Q12
    Q2 -->|Sí, falla| F2 --> Q12
    Q2 -.->|No, pasa OK| Q12
    Q3 -->|Sí, falla| F3 --> Q12
    Q3 -.->|No, pasa OK| Q12
    Q4 -->|Sí, falla| F4 --> Q12
    Q4 -.->|No, pasa OK| Q12
    Q5 -->|Sí, falla| F5 --> Q12
    Q5 -.->|No, pasa OK| Q12
    Q6 -->|Sí, falla| F6 --> Q12
    Q6 -.->|No, pasa OK| Q12
    Q7 -->|Sí, falla| F7 --> Q12
    Q7 -.->|No, pasa OK| Q12
    Q8 -->|Sí, falla| F8 --> Q12
    Q8 -.->|No, pasa OK| Q12
    Q9 -->|Sí, falla| F9 --> Q12
    Q9 -.->|No, pasa OK| Q12
    Q10 -->|Sí, falla| F10 --> Q12
    Q10 -.->|No, pasa OK| Q12
    Q11 -->|Sí, asimétrico| F11
    Q11 -.->|No, simétrico| Q12

    Q12 -->|Sí, al menos 1 FLAG| R1
    Q12 -->|No, todo OK| Q13
    Q13 -->|Sí, también se descarta| R1
    Q13 -->|No, se conserva| R2

    class A entrada
    class Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12,Q13 decision
    class F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11 marcado
    class R1 descartado
    class R2 salida
""" + _MERMAID_ESTILOS

_MERMAID_OE2 = """flowchart TD
    A["💾 Entrada: datos de control de calidad<br/>5 fuentes con flags + usabilidad"]

    subgraph FASE1["① EMPAREJAMIENTO — prepare.emparejar_horario_generico()"]
        B["🔗 Emparejamiento horario<br/>Solcast = referencia (variables predictoras)"]
        C1["📚 Conjunto SUPERVISADO<br/>horas medido_ok, cobertura mínima<br/>(para entrenar/evaluar)"]
        C2["🌐 Conjunto APLICACIÓN<br/>TODAS las horas con Solcast válido<br/>(para reconstruir la serie completa)"]
    end

    subgraph FASE2["② ENTRENAMIENTO Y VALIDACIÓN — prepare.split_temporal() · evaluate.entrenar_y_evaluar() · evaluate.elegir_mejor_modelo()"]
        D["✂️ Split temporal 70/30<br/>(split_temporal)"]
        E["🤖 Entrenar 9 modelos SOLO en TRAIN<br/>Baseline · Reg. simple · Reg. multivariable<br/>EQM · Random Forest · Extra Trees · Gradient Boosting<br/>XGBoost · Red neuronal (MLP)<br/>(entrenar_y_evaluar)"]
        F["📊 Validar SOLO en TEST<br/>(nunca visto en entrenamiento)<br/>MAE · RMSE · MBE · rMBE · rRMSE · Pearson R · R²"]
        G{{"🏆 Elegir modelo ganador<br/>1) físicamente válido en todo el período de aplicación<br/>2) menor RMSE<br/>(elegir_mejor_modelo)"}}
    end

    subgraph FASE3["③ RECONSTRUCCIÓN — evaluate.construir_serie_corregida()"]
        H["🔧 Aplicar el modelo YA entrenado<br/>sobre TODA la aplicación (no se reentrena)"]
        I1["Variante TOTAL<br/>reemplaza TODA la serie por la predicción"]
        I2["Variante SELECTIVA<br/>conserva medido_ok; corrige/rellena el resto"]
        J[("💾 Salida — site-adaptation<br/>serie corregida por estación")]
    end

    A --> B
    B --> C1
    B --> C2
    C1 --> D --> E --> F --> G --> H
    C2 --> H
    H --> I1 --> J
    H --> I2 --> J

    class A entrada
    class B,C1,C2,D,E,F,H proceso
    class G decision
    class I1,I2 variante
    class J salida
""" + _MERMAID_ESTILOS

_MERMAID_OE3 = """flowchart TD
    A["💾 Entrada: series adaptadas<br/>crudo · corregido_total · corregido_selectivo"]
    B["🔁 Re-aplicar QC físico + experto<br/>MISMAS pruebas de la etapa de control de calidad, serie por serie"]
    C["✅ Recalcular usabilidad por serie"]
    D["📊 Comparar % usable<br/>crudo vs. corregido_total vs. corregido_selectivo"]
    E["📈 Comparar métricas de desempeño<br/>Solcast crudo vs. adaptado, contra GHI real de la estación"]
    F["🔍 Detalle de banderas por prueba<br/>qué test causó cada descarte"]
    G[("💾 Salida — verificación final<br/>'dataset definitivo' = adaptado que SOBREVIVE al re-QC<br/>(subconjunto de la reconstrucción)")]

    A --> B --> C
    C --> D --> F
    C --> E --> F
    F --> G

    class A entrada
    class B,C,D,E,F proceso
    class G salida
""" + _MERMAID_ESTILOS


def render_oe1_diagrama_flujo():
    """Renderiza la pestaña 'Diagrama de flujo' de OE1: pipeline de control
    de calidad en Mermaid, con las condiciones de cada prueba y no solo
    una secuencia lineal."""
    seccion("Diagrama de flujo — OE1 (Control de Calidad)")
    tooltip("""
    Este es un diagrama de flujo real: cada rombo es una pregunta de
    sí/no (una prueba de QC individual), con el nombre exacto de la
    función que la implementa en `qcsolar/qc/physical.py` y
    `qcsolar/qc/expert.py`.

    **Importante — esto NO es una cadena/tubería.** Las 11 pruebas salen
    todas directamente del mismo dato crudo original (flechas desde el
    nodo de entrada hacia cada rombo): cada una lo evalúa **por su
    cuenta**, en paralelo, sin depender de lo que dijeron las demás. Que
    una prueba falle no "reprocesa" el dato en la siguiente prueba ni lo
    modifica — el dato que ve `test_nocturno()` es exactamente el mismo
    que vio `test_negativo()`, marcada o no la prueba anterior.

    **Cómo leer cada rombo:** si la prueba falla (Sí), el dato pasa por
    una casilla roja que dice explícitamente qué columna quedó marcada
    (p. ej. `flag_neg = FLAG`); si pasa (No, línea punteada), no hay
    ninguna consecuencia — simplemente esa prueba no aporta ninguna
    marca. Ambos caminos de cada prueba llegan al mismo punto de
    encuentro: la decisión final.

    Esa decisión final es el **único** lugar donde todas las pruebas se
    combinan: `calcular_usabilidad()` pregunta si **alguna** de las 10
    pruebas evaluables quedó en FLAG (si sí, el dato se descarta como
    NO_USABLE), y opcionalmente `aplicar_regla_islas()` descarta también
    un dato OK aislado entre dos tramos malos separados por menos de 60
    minutos. El resultado final es el dataset que **pasó** el control de
    calidad (nodo verde "OK"), tal como se explica en la pestaña **Marco
    metodológico** de este menú.

    Un caso especial es la **simetría AM/PM** (Q11): se evalúa igual que
    las demás y también se marca su flag si falla, pero por diseño (ver
    Marco metodológico) esa marca **no** llega a `calcular_usabilidad()`
    — por eso en el diagrama su flecha de "Sí" no se conecta a la
    decisión final: es diagnóstico, no excluyente.
    """)
    render_mermaid(_MERMAID_OE1, alto=1550)


def render_oe2_diagrama_flujo():
    """Renderiza la pestaña 'Diagrama de flujo' de OE2: pipeline de
    reconstrucción inversa en Mermaid (emparejamiento, partición,
    entrenamiento de los 9 modelos, selección del ganador y variantes
    total/selectiva de la serie corregida)."""
    seccion("Diagrama de flujo — OE2 (Site-Adaptation)")
    tooltip("""
    Este diagrama sintetiza el pipeline de reconstrucción inversa, el mismo
    que se explica modelo por modelo en la pestaña **Marco metodológico** de
    este menú. El conjunto *aplicación* (todas las horas) y el *supervisado*
    (solo horas confiables) se separan desde el emparejamiento — por eso el
    modelo se entrena/evalúa con uno y se aplica sobre el otro.

    **No es una simple predicción X→Y**: son 3 fases separadas, implementadas
    en funciones distintas del paquete `qcsolar`, cada una con su propio
    archivo fuente:

    - **① Emparejamiento** — `qcsolar/siteadapt/prepare.py`,
      función `emparejar_horario_generico()`.
    - **② Entrenamiento y validación** — también en `prepare.py`
      (`split_temporal()`, el 70/30), y en `qcsolar/siteadapt/evaluate.py`
      (`entrenar_y_evaluar()` entrena los 9 modelos SOLO en train y los
      evalúa SOLO en test; `elegir_mejor_modelo()` aplica el filtro de
      validez física + menor RMSE).
    - **③ Reconstrucción** — también en `evaluate.py`,
      función `construir_serie_corregida()`: toma el modelo YA entrenado
      (no lo vuelve a entrenar) y lo aplica sobre las 7.278 horas completas
      de Solcast para generar las 2 variantes (total/selectiva).

    Las 3 fases se ejecutan en secuencia, para las 4 estaciones, en el
    script `verificar_pieza7b_reconstruccion_4estaciones.py` — ese es el
    punto de entrada reproducible de todo el proceso.
    """)
    render_mermaid(_MERMAID_OE2, alto=680)


def render_oe3_diagrama_flujo():
    """Renderiza la pestaña 'Diagrama de flujo' de OE3: protocolo de
    verificación en Mermaid (re-QC de las 3 series, recálculo de
    usabilidad y comparación de % usable y métricas de error)."""
    seccion("Diagrama de flujo — OE3 (Verificación)")
    tooltip(
        "Este diagrama sintetiza el protocolo de verificación, el mismo que "
        "se explica en detalle en la pestaña **Marco metodológico** de este "
        "menú. Esta etapa no inventa pruebas nuevas: reutiliza el MISMO "
        "control de calidad de la etapa inicial, aplicado por separado a "
        "cada una de las 3 series (crudo, corregido total, corregido "
        "selectivo) para poder compararlas en igualdad de condiciones."
    )
    render_mermaid(_MERMAID_OE3, alto=480)


def render_oe3_resumen_4estaciones():
    """Renderiza la conclusión de OE3 para la estación terrestre: % usable
    por etapa, modelo ganador y RMSE por etapa. Debe invocarse solo dentro
    de la pestaña 'Evolución completa'; fuera de un bloque de pestaña,
    Streamlit la mostraría repetida en todas las pestañas."""
    st.markdown("<br>", unsafe_allow_html=True)
    _tablas_resumen_oe3 = cargar_reconstruccion_tablas()
    _df_comp_oe3 = _tablas_resumen_oe3["comparativo"]
    _df_resumen_oe3 = _tablas_resumen_oe3["resumen"]
    _df_metricas_oe3 = _tablas_resumen_oe3["metricas"]
    estaciones_oe3 = [("udenar", "Udenar (terrestre)", C_RED),
                      (CLAVE_UDENAR_EST, NOMBRE_UDENAR_EST, C_UDENAR_EST)]
    if _df_comp_oe3 is None or _df_comp_oe3.empty:
        st.info("No hay tabla comparativa todavía.")
        return

    seccion("Conclusión de la verificación")

    _nombres_est = {"udenar": "Udenar (terrestre)", CLAVE_UDENAR_EST: NOMBRE_UDENAR_EST}
    _variante_resumen = st.radio(
        "Variante a mostrar en el gráfico:", ["total", "selectivo"],
        horizontal=True, key="oe3_resumen_variante",
        format_func=lambda v: "Corregido — total" if v == "total" else "Corregido — selectivo",
    )
    _dfc_var = _df_comp_oe3[_df_comp_oe3["variante"] == _variante_resumen].copy()
    _dfc_var["nombre"] = _dfc_var["estacion"].map(_nombres_est)

    _dfc_var["pct_con_dato"] = _dfc_var["pct_medido_ok"] + _dfc_var["pct_medido_flag_corregido"]

    _fig_resumen = go.Figure()
    _fig_resumen.add_trace(go.Bar(
        name="① Con dato (antes de QC)", x=_dfc_var["nombre"], y=_dfc_var["pct_con_dato"],
        marker_color="#b9c4c9", opacity=0.9,
    ))
    _fig_resumen.add_trace(go.Bar(
        name="② Usable tras QC (crudo)", x=_dfc_var["nombre"], y=_dfc_var["pct_usable_crudo"],
        marker_color="#4d4d4d", opacity=0.85,
    ))
    _fig_resumen.add_trace(go.Bar(
        name=f"③ Usable tras QC del dato adaptado ({_variante_resumen})", x=_dfc_var["nombre"], y=_dfc_var["pct_usable_corregido"],
        marker_color=COLOR_CORREGIDO_TOTAL if _variante_resumen == "total" else COLOR_CORREGIDO_SELECTIVO,
        opacity=0.9,
    ))
    _fig_resumen.update_layout(
        barmode="group", height=380, plot_bgcolor="white", paper_bgcolor="white",
        title=f"% de datos físicamente utilizables — con dato vs. tras QC (crudo) vs. tras QC (adaptado, {_variante_resumen}), por estación",
        legend=dict(orientation="h", y=1.1, x=0),
        margin=dict(t=60, b=20, l=40, r=20),
        yaxis=dict(title="% usable", gridcolor="#f0f0f0", range=[0, 100]),
    )
    st.plotly_chart(_fig_resumen, use_container_width=True, key="oe3_resumen_4estaciones")
    panel_info(
        "Para cada estación: ① qué % de horas tiene alguna medición (antes de "
        "cualquier control de calidad), ② de esas, cuántas pasan el control de "
        "calidad aplicado al dato **crudo** tal cual llega, y ③ qué % pasa ese MISMO "
        "control de calidad ya aplicado al dato **adaptado** (corregido con Solcast "
        "como referencia y rellenado) — es decir, ③ ya es 'post-QC', solo que del dato "
        "adaptado en vez del crudo. No existe una barra '③ sin QC' porque no tendría "
        "sentido: el modelo rellena literalmente todas las horas, así que sin aplicar "
        "el control de calidad esa barra sería ~100% por construcción y no diría nada. "
        "La ganancia real que aporta el site-adaptation es ②→③ (post-QC del crudo vs. "
        "post-QC del adaptado); ① es solo el punto de partida (cuánto dato hay, sin "
        "decir si es bueno)."
    )

    _dfc_show = _df_comp_oe3[["estacion", "variante", "modelo_ganador",
                               "RMSE_test", "mejora_pct_usable_pp"]].copy()
    _dfc_show["estacion"] = _dfc_show["estacion"].map(_nombres_est)
    _dfc_show.columns = ["Estación", "Variante", "Modelo",
                          "RMSE test [W/m²]", "Mejora % usable [pp]"]
    subseccion("Modelo ganador y su desempeño, por estación")
    tabla_html(_dfc_show)
    boton_descarga_csv(_dfc_show, "oe3_resumen_4estaciones.csv")
    panel_info(
        "El modelo ganador de cada estación (columna Modelo) y su **RMSE test**: el "
        "error de esa predicción frente al GHI **real medido** en el conjunto de "
        "prueba (no frente a Solcast — Solcast es la entrada del modelo, no la "
        "referencia de este número). Aparece igual en 'total' y 'selectivo' porque "
        "ambas variantes usan el mismo modelo entrenado; solo difieren en qué hacen "
        "con las horas donde ya había una medición válida (ver pestaña de cada "
        "estación). **Mejora % usable [pp]**: cuántos puntos porcentuales sube el % "
        "de datos físicamente utilizables al pasar del dato crudo (② en el gráfico de "
        "arriba) a esa variante corregida (③) — el color de fondo identifica cada "
        "estación. El proceso completo (control de calidad → site-adaptation → "
        "verificación) produce, para cada estación, un dato solar corregido, "
        "trazable y físicamente defendible — el modelo ganador puede variar de una "
        "estación a otra."
    )

    # ── RMSE por etapa (crudo → post-QC → site-adaptado → +re-QC) ───────
    subseccion("RMSE por etapa — crudo vs. post-QC vs. site-adaptado")
    tooltip(
        "Ojo con la referencia: cambia según la etapa. **① Crudo** y **② Post-QC** "
        "comparan el dato terrestre contra **Solcast** (no hay otra verdad de terreno "
        "todavía). **③ Site-adaptado** y **④ +re-QC** comparan la predicción del modelo "
        "contra el **GHI real medido** en la estación (la verdad de terreno), en el "
        "mismo conjunto de prueba — por eso el RMSE de ③ NO es 'modelo vs. Solcast', es "
        "'modelo vs. estación real'. ④ usa la variante *total* porque *selectiva* ya "
        "conserva el dato medido donde era válido — comparar esa parte contra sí misma "
        "no diría nada nuevo."
    )

    _filas_rmse_etapa = []
    for _clave_e, _nombre_e, _ in estaciones_oe3:
        if _df_resumen_oe3 is None or _clave_e not in _df_resumen_oe3["estacion"].values:
            continue
        _fila_e = _df_resumen_oe3[_df_resumen_oe3["estacion"] == _clave_e].iloc[0]
        _rango_e = _parsear_rango_test(_fila_e.get("rango_test", ""))
        if _rango_e is None:
            continue
        _ref_e = calcular_metricas_referencia_ventana(_clave_e, _rango_e[0], _rango_e[1])
        if _ref_e is not None:
            for _etq, _label in [("crudo", "① Crudo (vs. Solcast)"), ("post_qc", "② Post-QC (vs. Solcast)")]:
                _m = _ref_e.get(_etq)
                if _m is not None and not pd.isna(_m.get("RMSE")):
                    _filas_rmse_etapa.append({"Estación": _nombre_e, "Etapa": _label,
                                               "RMSE [W/m²]": round(_m["RMSE"], 2)})
        _dfm_g_e = _df_metricas_oe3[(_df_metricas_oe3["estacion"] == _clave_e)
                                     & (_df_metricas_oe3["modelo_ganador"])] if _df_metricas_oe3 is not None else None
        if _dfm_g_e is not None and not _dfm_g_e.empty:
            _filas_rmse_etapa.append({
                "Estación": _nombre_e,
                "Etapa": f"③ Site-adaptado, sin re-QC ({_dfm_g_e.iloc[0]['modelo']})",
                "RMSE [W/m²]": round(_dfm_g_e.iloc[0]["RMSE"], 2),
            })
        _reqc_e = (calcular_rmse_adaptado_tras_reqc(_clave_e, _rango_e[0], _rango_e[1])
                   if _rango_e is not None else None)
        if _reqc_e is not None and _reqc_e.get("con_reqc") is not None:
            _filas_rmse_etapa.append({"Estación": _nombre_e, "Etapa": "④ Site-adaptado, con re-QC (total)",
                                       "RMSE [W/m²]": round(_reqc_e["con_reqc"]["RMSE"], 2)})

    if not _filas_rmse_etapa:
        st.info("No hay datos suficientes para el desglose de RMSE por etapa.")
        return

    _df_rmse_etapa = pd.DataFrame(_filas_rmse_etapa)
    _df_rmse_etapa["EtapaGrupo"] = _df_rmse_etapa["Etapa"].str.extract(r"^([①②③④])")
    _colores_etapa_oe3 = {"①": "#b9c4c9", "②": C_GOLD, "③": "#8fb8d6", "④": C_AZUL_VIVO}
    _fig_rmse_etapa = go.Figure()
    for _grp in ["①", "②", "③", "④"]:
        _sub_e = _df_rmse_etapa[_df_rmse_etapa["EtapaGrupo"] == _grp]
        if _sub_e.empty:
            continue
        _fig_rmse_etapa.add_trace(go.Bar(
            name=_sub_e["Etapa"].iloc[0].split(" (")[0], x=_sub_e["Estación"], y=_sub_e["RMSE [W/m²]"],
            marker_color=_colores_etapa_oe3[_grp],
            text=[f"{v:.1f}" for v in _sub_e["RMSE [W/m²]"]], textposition="outside",
            customdata=_sub_e["Etapa"], hovertemplate="%{x}<br>%{customdata}: %{y:.1f} W/m²<extra></extra>",
        ))
    _fig_rmse_etapa.update_layout(
        barmode="group", height=380, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=40, b=40, l=40, r=20),
        yaxis=dict(title="RMSE [W/m²]", gridcolor="#f0f0f0"),
    )
    st.plotly_chart(_fig_rmse_etapa, use_container_width=True, key="oe3_rmse_por_etapa")

    _df_rmse_show = _df_rmse_etapa.drop(columns=["EtapaGrupo"])
    boton_descarga_csv(_df_rmse_show, "oe3_rmse_por_etapa.csv")
    panel_info(
        "Lo esperable es una escalera descendente ①→②→③, y que ④ quede muy cerca "
        "de ③ (el re-QC final debería descartar pocas o ninguna predicción, ya que "
        "los modelos ganadores no pueden extrapolar fuera de rango). Si ④ es "
        "notablemente menor que ③, significa que el re-QC sí está filtrando algunas "
        "predicciones malas puntuales; si son iguales, el modelo ya era consistente "
        "sin necesitar ese filtro adicional."
    )


def _render_udenar_oe1(sel_udenar_oe1):
    """Renderiza la sub-pestaña de la estación terrestre en OE1 para el caso
    seleccionado (el argumento es la etiqueta del año). Se separa en una
    función para mantener corta la rama correspondiente de la página OE1."""
    _map_n2 = {"udenar": "Udenar (terrestre)", CLAVE_UDENAR_EST: NOMBRE_UDENAR_EST}
    _todas_udenar_oe1 = [
        ("udenar", "Udenar (terrestre)", "2023"),
        (CLAVE_UDENAR_EST, NOMBRE_UDENAR_EST, "2025"),
    ]
    estaciones = [(c, n, st.container()) for c, n, a in _todas_udenar_oe1 if sel_udenar_oe1 == f"📅 {a}"]
    for clave, nombre, tab in estaciones:
        with tab:
            seccion(f"{nombre} — Estación terrestre")

            _est_n2 = _map_n2.get(clave, nombre)
            _est_s2 = cargar_crudos_global().get(_est_n2)
            _est_geo = cargar_procesado(f"{clave}_qc_completo.csv")
            _est_geo_postqc = _construir_df_etapa_diagnostico(clave, "post_qc")

            if _est_s2 is not None:
                st.plotly_chart(
                    fig_cobertura_mensual_plotly(_est_s2, nombre, COLORES_FUENTES.get(_est_n2, C_TEAL)),
                    use_container_width=True, key=f"cob_mensual_{clave}",
                )
                panel_info(
                    "Cuántos registros hay por mes frente a lo esperado a la resolución nativa "
                    "de esta estación — barras cortas o ausentes indican meses con huecos de cobertura."
                )
                _res_est = resumen_intervalo_serie(_est_s2)
                _cm1, _cm2, _cm3 = st.columns(3)
                _cm1.metric("Intervalo típico", f"{_res_est['intervalo_min']:.1f} min" if _res_est["intervalo_min"] else "—")
                _cm2.metric("N registros", f"{_res_est['n_registros']:,}")
                _cm3.metric("Período", _res_est["periodo"])
            else:
                mostrar_figura(f"00_{clave}_cobertura_mensual.png", f"Cobertura mensual — {nombre}")
                boton_descarga_fig(f"00_{clave}_cobertura_mensual.png")

            if _est_s2 is not None:
                st.plotly_chart(
                    fig_continuidad_temporal_plotly(_est_s2, nombre, COLORES_FUENTES.get(_est_n2, C_TEAL)),
                    use_container_width=True, key=f"continuidad_{clave}",
                )
                panel_info(
                    "Mapa día × hora de disponibilidad de datos — franjas claras u oscuras "
                    "marcan huecos concentrados en ciertas fechas u horas del día."
                )
            else:
                mostrar_figura(f"00_{clave}_continuidad_temporal.png",
                               f"Continuidad temporal (día × hora) — {nombre}. "
                               "Franjas claras = huecos de datos.")
                boton_descarga_fig(f"00_{clave}_continuidad_temporal.png")

            subseccion("Datos marcados por control de calidad")
            tooltip(
                "Este gráfico usa un umbral SIMPLIFICADO e ilustrativo (no las pruebas "
                "físicas/expertas exactas del proyecto — para esas ver las tablas de % de "
                "flag en el tab Resumen). Colores: azul con relleno de fondo=crudo (el "
                "100% de los registros), verde=OK (dentro de límites físicos aproximados), "
                "rojo/naranja=FLAG (fuera de límites o nocturno con irradiancia positiva), "
                "gris=CERO/NEGATIVO.\n\n"
                "**Importante sobre 'CERO/NEGATIVO':** casi todo ese grupo es GHI = 0 "
                "**de noche** — algo esperado y normal (de noche no hay radiación), no un "
                "defecto del sensor. No lo confundas con un dato malo: la proporción de "
                "noche/día en un año es, por diseño, cercana a 50/50.\n\n"
                "Los límites exactos dependen del ángulo cenital (pvlib). "
                "Aquí se usa un umbral simplificado: GHI > 1100 W/m² → FLAG probable, "
                "GHI nocturno (valor alto con hora fuera de rango solar) → FLAG nocturno."
            )
            if _est_s2 is not None:
                _s2 = _est_s2.dropna().reset_index()
                _s2.columns = ["dt", "ghi"]
                # Clasificación simple: OK / FLAG_ALTO / FLAG_NOCTURNO / CERO
                _s2["hora"] = _s2["dt"].dt.hour
                _s2["tipo"] = "OK"
                _s2.loc[_s2["ghi"] <= 0, "tipo"] = "CERO/NEGATIVO"
                _s2.loc[(_s2["ghi"] > 0) & (_s2["ghi"] > 1100), "tipo"] = "FLAG (límite físico)"
                # Nocturno: horas fuera de 5-19 con GHI > 10
                _s2.loc[(_s2["hora"] < 5) & (_s2["ghi"] > 10), "tipo"] = "FLAG (nocturno)"
                _s2.loc[(_s2["hora"] >= 19) & (_s2["ghi"] > 10), "tipo"] = "FLAG (nocturno)"
                _colores_tipo = {
                    "OK":                  "#22AA44",
                    "CERO/NEGATIVO":       "#CCCCCC",
                    "FLAG (límite físico)":"#E63946",
                    "FLAG (nocturno)":     "#FF8800",
                }
                _total_s2 = len(_s2)
                _fig_flag = go.Figure()
                _fig_flag.add_trace(go.Scattergl(
                    x=_s2["dt"], y=_s2["ghi"],
                    mode="lines",
                    line=dict(color=C_AZUL_VIVO, width=1),
                    fill="tozeroy", fillcolor="rgba(21, 101, 192, 0.15)",
                    opacity=0.8,
                    name=f"Crudo, 100% ({_total_s2:,})",
                    hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI crudo: %{y:.1f} W/m²<extra></extra>",
                ))
                for _tipo, _color in _colores_tipo.items():
                    _mask = _s2["tipo"] == _tipo
                    if _mask.sum() == 0:
                        continue
                    _sub = _s2[_mask]
                    _pct_tipo = 100 * _mask.sum() / _total_s2 if _total_s2 else 0
                    _fig_flag.add_trace(go.Scattergl(
                        x=_sub["dt"], y=_sub["ghi"],
                        mode="lines",
                        line=dict(color=_color, width=1),
                        opacity=0.65 if _tipo == "OK" else 0.9,
                        name=f"{_tipo} ({_mask.sum():,} · {_pct_tipo:.1f}%)",
                        hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.1f} W/m²<extra></extra>",
                    ))
                _fig_flag.update_layout(
                    title=f"{nombre} — GHI coloreado por calidad",
                    height=420, plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(
                        orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                        font=dict(size=12), bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="#ccc", borderwidth=1,
                        itemsizing="constant",
                    ),
                    margin=dict(t=50, b=20, l=60, r=170),
                    xaxis=dict(showgrid=False, tickformat="%b %Y",
                        rangeselector=dict(buttons=[
                            dict(count=7,  label="1S",  step="day",   stepmode="backward"),
                            dict(count=1,  label="1M",  step="month", stepmode="backward"),
                            dict(count=3,  label="3M",  step="month", stepmode="backward"),
                            dict(step="all", label="Todo"),
                        ], bgcolor="#f0f6f9", activecolor="#1F4E5F", font=dict(size=10)),
                        rangeslider=dict(visible=True, thickness=0.07),
                    ),
                    yaxis=dict(title=dict(text="GHI [W/m²]", font=dict(size=11)),
                               gridcolor="#eeeeee", rangemode="tozero"),
                )
                st.plotly_chart(_fig_flag, use_container_width=True,
                                key=f"flag_ghi_{clave}")
                # Resumen de conteos
                _conteos = _s2["tipo"].value_counts()
                _total = len(_s2)
                _cols_flag = st.columns(len(_conteos))
                for _ci, (_tipo, _cnt) in enumerate(_conteos.items()):
                    _cols_flag[_ci].metric(
                        label=_tipo,
                        value=f"{_cnt:,}",
                        delta=f"{100*_cnt/_total:.1f}%",
                        delta_color="off",
                    )
            else:
                mostrar_figura(f"11_{clave}_ghi_flagueado.png",
                               f"GHI flagueado — {nombre}")
                boton_descarga_fig(f"11_{clave}_ghi_flagueado.png")

            mostrar_figura(f"12_{clave}_flags_por_mes.png", f"Flags físicos y expertos por mes — {nombre}")
            boton_descarga_fig(f"12_{clave}_flags_por_mes.png")
            panel_info(
                "% de datos marcados por cada prueba de QC, mes a mes — picos puntuales "
                "suelen ser eventos climáticos reales, picos sostenidos indican un problema "
                "del sensor."
            )

            _fig_csi_cmp = fig_csi_comparativo_plotly(_est_geo, _est_geo_postqc, nombre)
            if _fig_csi_cmp is not None:
                st.plotly_chart(_fig_csi_cmp, use_container_width=True, key=f"csi_anual_cmp_{clave}")
            else:
                mostrar_figura(f"14_{clave}_csi_anual.png", f"Índice de cielo claro (CSI) — {nombre}")
                boton_descarga_fig(f"14_{clave}_csi_anual.png")
            panel_info(
                "CSI diario mediano, crudo (rojo) vs. post-QC (azul) — si el QC funciona "
                "bien, la curva azul debería verse más centrada alrededor de 1 y sin los "
                "picos extremos que sí tiene la curva roja."
            )

            _fig_sim_cmp = fig_simetria_comparativo_plotly(_est_geo, _est_geo_postqc, nombre)
            if _fig_sim_cmp is not None:
                st.plotly_chart(_fig_sim_cmp, use_container_width=True, key=f"simetria_azimut_cmp_{clave}")
            else:
                mostrar_figura(f"19_simetria_azimut_{clave}.png",
                               f"Simetría AM/PM vs azimut solar — {nombre}")
                boton_descarga_fig(f"19_simetria_azimut_{clave}.png")

            tooltip("""
            **Prueba de simetría AM/PM** — Para cada día se compara la energía
            acumulada en la mañana (AM) contra la tarde (PM). Una asimetría
            sistemática (>15%) puede indicar: obstrucción del horizonte, sensor mal
            nivelado, o el patrón climático real de Pasto (nubosidad convectiva
            de tarde). En rojo, con todos los datos (crudo); en azul, solo con
            las horas que pasan el control de calidad (post-QC) — los días con
            medios días caídos o lecturas erráticas (que pueden generar asimetrías
            artificiales de ±100%) quedan filtrados en la serie azul. El gráfico
            muestra que la asimetría **no tiene preferencia direccional** — se
            distribuye igual en azimuts Norte y Sur, consistente con un forzante
            climático, no instrumental.
            """)

            subseccion(f"Patrón anual día × hora — GHI ({nombre})")
            _fig_hm_cmp = fig_heatmap_comparativo_plotly(
                (_est_geo, _est_geo_postqc), ("① Crudo", "② Post-QC"), "GHI", nombre)
            if _fig_hm_cmp is not None:
                st.plotly_chart(_fig_hm_cmp, use_container_width=True, key=f"heatmap_cmp_{clave}")
                panel_info(
                    "Mapa de calor día del año × hora local: cada franja horizontal es un día "
                    "completo del año. Las 2 versiones comparten la MISMA escala de color, para "
                    "que sean directamente comparables. Huecos o franjas en blanco que aparecen "
                    "solo en post-QC (② vs. ①) son las horas que el control de calidad descarta."
                )
            else:
                st.info(f"No hay datos suficientes de {nombre} para este panel.")

            subseccion(f"Envolvente solar — mapa elevación × azimut ({nombre})")
            _fig_env_cmp = fig_envolvente_comparativo_plotly(
                (_est_geo, _est_geo_postqc), ("① Crudo", "② Post-QC"), "GHI", nombre)
            if _fig_env_cmp is not None:
                st.plotly_chart(_fig_env_cmp, use_container_width=True, key=f"envolvente_cmp_{clave}")
            else:
                mostrar_figura(f"22_envolvente_ghi_{clave}.png",
                               f"{nombre} — envolvente máxima de GHI")
                boton_descarga_fig(f"22_envolvente_ghi_{clave}.png")
            panel_info(
                "Para cada celda del cielo (elevación × azimut solar) se pinta el valor máximo "
                "de GHI registrado en todo el periodo. Las 2 versiones comparten la MISMA escala "
                "de color, para que sean directamente comparables. Una celda que nunca alcanza el "
                "valor esperado para esa elevación (un \"valle oscuro\" que no es el borde del "
                "arco) sugiere obstrucción del horizonte en esa dirección — si ese valle persiste "
                "igual en las 2 versiones, es evidencia de obstrucción real (no un artefacto de "
                "datos sucios que el QC ya habría descartado). Un arco fragmentado en vez de "
                "continuo indica huecos de cobertura temporal. La forma de \"dos arcos\" es normal "
                "para Pasto: al estar casi sobre el ecuador (1.21°N), el sol pasa al norte en "
                "junio y al sur en diciembre, dando dos trayectorias distintas."
            )


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PÁGINA: INICIO  ██████████████████
# ─────────────────────────────────────────────────────────────────────────────
if "Inicio" in pagina:
    _sel_inicio = tabs_perezosas(["🏠 Resumen", "📐 Marco metodológico — Métricas"], key="tabs_inicio_sel")
    if _sel_inicio == "🏠 Resumen":
        render_inicio_resumen()
    elif _sel_inicio == "📐 Marco metodológico — Métricas":
        render_inicio_marco_metricas()

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PÁGINA: OE1  ██████████████████
# ─────────────────────────────────────────────────────────────────────────────
elif "OE1" in pagina:
    st.markdown(
        f"""<div style='background:linear-gradient(135deg,{C_TEAL},{C_BLUE});
        color:white; padding:28px 36px; border-radius:16px; margin-bottom:1.5rem'>
        <div style='font-size:2.25rem; font-weight:600; color:white; margin:0'>🔬 OE1 — Control de Calidad</div>
        <p style='color:#d0eaf4; margin-top:6px; font-size:1rem'>
        Protocolo de control de calidad (estándar BSRN) · 2 casos: Udenar 2023 y
        {NOMBRE_UDENAR_EST}</p></div>""",
        unsafe_allow_html=True,
    )

    tooltip("""
    **¿Qué se hizo en esta etapa de control de calidad?**

    Se aplicó un protocolo de control de calidad basado en el estándar BSRN
    a las 2 fuentes de irradiancia disponibles. El protocolo tiene tres capas:

    - **Pruebas físicas (BSRN):** detectan valores imposibles físicamente
      (negativos, sobre el límite superior, nocturnos con irradiancia).
    - **Pruebas de experto:** cierre radiativo (GHI = DNI·cos(SZA) + DHI),
      K-tests (índices de claridad), índice de cielo claro (CSI), simetría AM/PM,
      tasa de cambio y persistencia.
    - **Usabilidad:** combina los flags anteriores para marcar cada dato como
      OK, FLAG o NO_EVALUABLE.

    Cada prueba devuelve por dato una de tres etiquetas:
    `OK` · `FLAG` · `NO_EVALUABLE`
    """)

    # ── TABS de OE1 ──────────────────────────────────────────────────────────
    _etq_oe1 = [
        "📐 Marco metodológico",
        "📊 Resumen comparativo",
        "📡 Solcast",
        "🌱 Udenar",
        "📈 Métricas cruzadas",
        "🧭 Diagrama de flujo",
    ]
    _sel_oe1 = tabs_perezosas(_etq_oe1, key="tabs_oe1_sel")

    # Cargar tablas QC
    df_fisico = cargar_csv("qc_fisico_flags_comparativo.csv", ROOT / "outputs" / "tablas")
    if df_fisico is None:
        df_fisico = pd.read_csv(ROOT / "qc_fisico_flags_comparativo.csv") if (ROOT / "qc_fisico_flags_comparativo.csv").exists() else None

    df_experto = cargar_csv("qc_experto_flags_comparativo.csv", ROOT / "outputs" / "tablas")
    if df_experto is None:
        df_experto = pd.read_csv(ROOT / "qc_experto_flags_comparativo.csv") if (ROOT / "qc_experto_flags_comparativo.csv").exists() else None

    df_usab = cargar_csv("usabilidad_comparativo.csv", ROOT / "outputs" / "tablas")
    if df_usab is None:
        df_usab = pd.read_csv(ROOT / "usabilidad_comparativo.csv") if (ROOT / "usabilidad_comparativo.csv").exists() else None

    # ── TAB 0: Marco metodológico ────────────────────────────────────────────
    if _sel_oe1 == _etq_oe1[0]:
        seccion("Marco metodológico — Protocolo de control de calidad (QC)")

        st.markdown("""
        El protocolo implementado sigue las prácticas estándar internacionales
        de control de calidad de datos de radiación solar, basadas en los
        límites físicos de la red **BSRN** (Baseline Surface Radiation Network).

        Cada registro recibe una de tres etiquetas por prueba:

        | Etiqueta | Significado |
        |---|---|
        | ✅ `OK` | El dato pasa la prueba |
        | 🚩 `FLAG` | El dato es sospechoso o imposible |
        | ⬜ `NO_EVALUABLE` | La prueba no aplica (ej: noche en pruebas diurnas) |

        Un dato se declara **NO_USABLE** si al menos una prueba evaluable lo marca FLAG.
        """)

        # ── FAMILIA 1: Pruebas físicas BSRN ──────────────────────────────────
        subseccion("Familia 1 — Pruebas físicas BSRN")
        st.markdown("""
        Verifican que los valores estén dentro de límites físicamente posibles.
        Los límites **no son constantes** — dependen de la posición del sol en
        cada instante, calculada con `pvlib` (ángulo cenital **SZA** e
        irradiancia extraterrestre **ETN**).
        """)

        with st.expander("📏 Prueba 1 — Límite físicamente posible (PPL)", expanded=True):
            col1, col2 = st.columns([1.1, 1])
            with col1:
                st.markdown("**Fórmula (límites físicamente posibles, estándar BSRN):**")
                st.latex(r"""
                -4 \leq GHI \leq 1.5 \cdot E_0 \cdot \cos^{1.2}(\theta_z) + 100
                \quad [\text{W/m}^2]
                """)
                st.markdown(r"""
                **Variables:**
                - $E_0$ = irradiancia extraterrestre ≈ 1361 W/m² × factor de distancia solar
                - $\theta_z$ = ángulo cenital solar (0° = sol en cénit, 90° = horizonte)
                - El exponente 1.2 modela la atenuación atmosférica con la masa de aire

                **En Udenar a mediodía solar típico:**
                - $\theta_z \approx 20.6°$ (latitud ~1.23°N, hora local mediodía)
                - $E_0 \approx 1321$ W/m²
                - Límite superior ≈ $1.5 × 1321 × \cos^{1.2}(20.6°) + 100 \approx$ **1931 W/m²**

                **Parámetros de configuración usados:**
                ```yaml
                ppl_ghi_c1: 1.5   # coeficiente multiplicador
                ppl_ghi_c2: 1.2   # exponente del coseno
                ppl_ghi_c3: 100.0 # constante aditiva [W/m²]
                minimo_irradiancia: -4.0  # límite inferior BSRN
                ```
                """)
            with col2:
                st.markdown("**Ejemplo real del proyecto:**")
                st.info("""
                📅 **15 jul 2023, 12:00 — Udenar**

                - SZA = 20.6°
                - E₀ = 1321 W/m²
                - GHI medido Udenar = **277.3 W/m²**
                - Límite PPL superior = 1.5 × 1321 × cos¹·²(20.6°) + 100 = **1931 W/m²**
                - Límite inferior = **−4 W/m²**

                ✅ **Resultado: OK**
                (277.3 está entre −4 y 1931)
                """)
                st.error("""
                📅 **16 jul 2023, 18:20 — Udenar** (crepúsculo)

                - GHI medido Udenar = **10.2 W/m²**
                - Elevación solar = −0.65° (sol ya bajo el horizonte)
                - La prueba nocturna (no la PPL) marca esta hora

                🚩 **Resultado: FLAG nocturno**
                (irradiancia positiva con sol bajo el horizonte)
                """)

        with st.expander("📏 Prueba 2 — Límite extremadamente raro (ERL)"):
            col1, col2 = st.columns([1.1, 1])
            with col1:
                st.markdown("**Fórmula (límites extremadamente raros, estándar BSRN):**")
                st.latex(r"""
                -2 \leq GHI \leq 1.2 \cdot E_0 \cdot \cos^{1.2}(\theta_z) + 50
                \quad [\text{W/m}^2]
                """)
                st.markdown("""
                Más restrictivo que PPL: marca datos que aunque son físicamente
                posibles, son estadísticamente muy raros. En la práctica, un dato
                que supera ERL pero no PPL puede ser un evento de
                **realce por nubes** (cloud enhancement, un fenómeno físico real
                y documentado en la literatura de radiación solar).

                **Parámetros de configuración usados:**
                ```yaml
                erl_ghi_c1: 1.2
                erl_ghi_c2: 1.2
                erl_ghi_c3: 50.0
                minimo_erl: -2.0
                ```
                """)
            with col2:
                st.markdown("**Resultado en el proyecto:**")
                st.info("""
                En Solcast: **0.0%** de los registros superan el
                límite ERL.

                En Udenar: **0.002%** — prácticamente nulo. A
                diferencia de estaciones con sensores más ruidosos,
                aquí el límite extremadamente raro casi no se activa;
                el modo de falla dominante en Udenar es otro (ver
                Simetría AM/PM más abajo), no valores físicamente
                extremos.
                """)

        with st.expander("📏 Prueba 3 — GHI nocturno"):
            col1, col2 = st.columns([1.1, 1])
            with col1:
                st.markdown("**Criterio:**")
                st.latex(r"""
                \text{FLAG si } GHI > 10 \text{ W/m}^2 \text{ y } \text{elevación solar} < 0°
                """)
                st.markdown("""
                Cuando el sol está bajo el horizonte (elevación < 0°), la
                irradiancia solar directa y difusa deben ser ≈ 0. Un valor
                superior a **10 W/m²** en esas condiciones indica un offset
                del sensor, ruido eléctrico o falla de calibración.

                **Parámetros de configuración usados:**
                ```yaml
                elevacion_noche_grados: 0.0
                max_irradiancia_noche: 10.0  # W/m²
                ```
                """)
            with col2:
                st.markdown("**Hallazgo en Udenar:**")
                st.info("""
                **0.094%** de los registros de Udenar tienen GHI
                nocturno por encima del umbral — un porcentaje bajo,
                sin el patrón de offset sistemático y prolongado que
                se documentó en el proyecto original con otra
                estación (cientos de noches consecutivas con offset
                constante). Aquí no fue necesario restringir la
                ventana de análisis por esta causa — la ventana real
                (jul-dic 2023) está acotada por la disponibilidad del
                dato del inversor, no por un problema de sensor
                nocturno.
                """)

        with st.expander("📏 Prueba 4 — Sensor caído"):
            st.markdown("""
            **Criterio:** valor constante durante más de **60 minutos consecutivos**
            en horas diurnas (elevación solar > 0°).

            Un piranómetro real nunca produce exactamente el mismo valor durante
            una hora de luz solar — la irradiancia varía con las nubes, el ángulo
            solar y la turbidez atmosférica. Un bloque de valores idénticos indica
            que el sensor se "congeló" (falla del datalogger, desconexión o falla
            del sensor).

            **Solo aplica a estaciones terrestres** — Solcast es un modelo y
            puede producir valores similares en períodos de cielo muy estable,
            lo cual no es una falla.

            ```yaml
            detectar_sensor_caido: true
            # Umbral: ventana de 60 minutos con std = 0
            ```
            """)

        st.markdown("---")

        # ── FAMILIA 2: Pruebas expertas ───────────────────────────────────────
        subseccion("Familia 2 — Pruebas de experto (solo Solcast)")

        st.markdown("""
        Estas pruebas requieren las **tres componentes de irradiancia**
        (GHI, DNI, DHI) y verifican la consistencia entre ellas.
        Solo aplican a **Solcast**, que reporta las tres.
        Las estaciones terrestres solo miden GHI.
        """)

        with st.expander("🔬 Prueba 5 — Cierre radiativo", expanded=True):
            col1, col2 = st.columns([1.1, 1])
            with col1:
                st.markdown("**Ecuación de cierre (base física):**")
                st.latex(r"""
                GHI = DNI \cdot \cos(\theta_z) + DHI
                """)
                st.markdown("**Criterio de flag (estándar BSRN):**")
                st.latex(r"""
                \left| \frac{GHI}{DNI \cdot \cos(\theta_z) + DHI} - 1 \right| > \varepsilon
                """)
                st.markdown(r"""
                donde la tolerancia ε depende del ángulo cenital:

                | SZA | Tolerancia ε |
                |---|---|
                | ≤ 75° | 8% |
                | > 75° | 15% |

                La tolerancia es mayor para ángulos altos porque la masa de
                aire atmosférica es mayor y las incertidumbres aumentan.

                **Solo se evalúa si GHI > 50 W/m²** (evitar divisiones por
                valores cercanos a cero al amanecer/atardecer).

                **Parámetros de configuración usados:**
                ```yaml
                cierre_ghi_min: 50.0
                cierre_tol_sza_baja: 0.08   # 8% para SZA ≤ 75°
                cierre_tol_sza_alta: 0.15   # 15% para SZA > 75°
                cierre_sza_corte: 75.0
                ```
                """)
            with col2:
                st.markdown("**Ejemplo numérico:**")
                st.success("""
                📅 **10 mar 2025, 10:30 — Solcast**

                - GHI = 650 W/m²
                - DNI = 720 W/m²
                - DHI = 95 W/m²
                - SZA = 23.4° → cos(23.4°) = 0.917

                **GHI calculado** = 720 × 0.917 + 95 = **755 W/m²**

                **Desviación** = |650/755 − 1| = **0.139 = 13.9%**

                Tolerancia para SZA ≤ 75° = **8%**

                🚩 **Resultado: FLAG** (13.9% > 8%)
                """)
                st.info("""
                **Resultado en Solcast (año 2025):**
                Solo el **0.0%** de los registros fallan
                el cierre radiativo — Solcast mantiene
                perfecta consistencia interna entre sus
                tres componentes por diseño.
                """)

        with st.expander("🔬 Prueba 6 — K-tests (índices de claridad)"):
            col1, col2 = st.columns([1.1, 1])
            with col1:
                st.markdown("**Definición de índices (estándar BSRN):**")
                st.latex(r"""
                K_t = \frac{GHI}{E_0 \cdot \cos(\theta_z)} \quad
                K_n = \frac{DNI}{E_0} \quad
                K = \frac{DHI}{GHI}
                """)
                st.markdown(r"""
                Estos índices representan la fracción del recurso solar
                disponible que llega a la superficie. Sus límites físicos
                forman una envolvente en el espacio ($K_t$, $K_n$, $K$):

                | Índice | Rango físico |
                |---|---|
                | $K_t$ (claridad GHI) | 0 ≤ Kt ≤ 1.35 |
                | $K_n$ (claridad DNI) | 0 ≤ Kn ≤ 1.0 |
                | $K$ (fracción difusa) | $K ≤ 1 − 0.1(K_t − 0.2)$ cuando Kt > 0.6 |

                Un valor fuera de esta envolvente indica inconsistencia
                física entre las componentes.

                **Solo se evalúa si GHI > 50 W/m².**
                """)
            with col2:
                st.markdown("**Resultado en Solcast:**")
                st.success("""
                **0% de flags en K-tests** para Solcast.

                Los tres índices permanecen dentro de la
                envolvente física durante todo el período 2023
                analizado, consistente con que Solcast es un modelo
                atmosférico que garantiza consistencia
                interna de sus componentes por diseño.

                Las figuras 21A y 21B (tab Solcast)
                muestran los diagramas Kn vs Kt y
                K vs Kt con los límites superpuestos.
                """)

        st.markdown("---")

        # ── FAMILIA 3: Pruebas de serie temporal ─────────────────────────────
        subseccion("Familia 3 — Pruebas de serie temporal (todas las fuentes)")

        st.markdown("""
        Estas pruebas no requieren DNI ni DHI — operan sobre la serie
        temporal de GHI únicamente. Se aplican a **todas las fuentes**.
        """)

        with st.expander("📈 Prueba 7 — Índice de cielo despejado (CSI)", expanded=True):
            col1, col2 = st.columns([1.1, 1])
            with col1:
                st.markdown("**Fórmula:**")
                st.latex(r"""
                CSI = \frac{GHI}{GHI_{cs}}
                """)
                st.markdown(r"""
                donde $GHI_{cs}$ es la irradiancia teórica de **cielo despejado**,
                calculada con el modelo Ineichen-Perez vía `pvlib` para las
                coordenadas exactas de Udenar (lat=1.233203°, lon=−77.292877°,
                alt=2451 m).

                **Criterio de flag:**
                $$CSI > CSI_{max} = 1.35$$

                Un CSI > 1 significa que la medición supera lo que habría con
                cielo completamente despejado — físicamente posible por
                **realce de nubes** (reflexión lateral), pero estadísticamente
                raro. Un CSI > 1.35 se considera anómalo.

                **Solo se evalúa si GHI > 50 W/m² y de día.**

                ```yaml
                csi_max: 1.35
                csi_ghi_min: 50.0
                ```
                """)
            with col2:
                st.markdown("**Ejemplo numérico (ilustrativo):**")
                st.warning("""
                📅 **Día nublado parcial (ejemplo ilustrativo)**

                - GHI medido = 980 W/m²
                - GHI cielo despejado (pvlib) = 820 W/m²
                - CSI = 980 / 820 = **1.195**

                ✅ **OK** (1.195 < 1.35)
                Posible realce leve por nubes, pero
                dentro del rango aceptable.
                """)
                st.error("""
                📅 **Evento extremo**

                - GHI medido = 1200 W/m²
                - GHI cielo despejado = 840 W/m²
                - CSI = 1200 / 840 = **1.429**

                🚩 **FLAG** (1.429 > 1.35)
                Realce por nubes excesivo o error
                de sensor.
                """)

        with st.expander("📈 Prueba 8 — Tasa de cambio (Rate of Change)"):
            col1, col2 = st.columns([1.1, 1])
            with col1:
                st.markdown("**Fórmula:**")
                st.latex(r"""
                ROC = \frac{|GHI_t - GHI_{t-1}|}{\Delta t_{min}}
                \quad [\text{W/m}^2/\text{min}]
                """)
                st.markdown(r"""
                donde $\Delta t_{min}$ es el intervalo entre muestras en minutos.

                **Criterio de flag:**
                $$ROC > ROC_{max}$$

                Un cambio de irradiancia excesivamente rápido entre dos
                muestras consecutivas indica un salto anómalo del sensor
                (descarga electrostática, interferencia, error de lectura).

                La variabilidad máxima natural del recurso solar en Udenar
                (paso rápido de nubes) se estimó en la calibración del umbral.

                **Solo de día.** Solo entre muestras del mismo día.

                ```yaml
                roc_max_wm2_min: 100.0  # W/m²/min
                ```
                """)
            with col2:
                st.markdown("**Resultado en el proyecto:**")
                st.info("""
                **Udenar:** **0.002%** de los registros marcados
                por tasa de cambio — prácticamente nulo.

                **Solcast:** **0.0%** — los valores
                satelitales no tienen discontinuidades
                abruptas por diseño del modelo.
                """)

        with st.expander("📈 Prueba 9 — Persistencia"):
            col1, col2 = st.columns([1.1, 1])
            with col1:
                st.markdown("**Criterio:**")
                st.latex(r"""
                \text{FLAG si} \geq 15 \text{ valores identicos consecutivos}
                \text{ de GHI, con el sol sobre el horizonte}
                """)
                st.markdown(r"""
                Se identifican rachas de GHI idéntico entre muestras
                consecutivas (a 5 min de resolución, 15 muestras ≈ 75 min).
                Si la racha alcanza 15 o más **mientras hay sol** (elevación
                solar > 0°), se marca FLAG — **sin importar si el valor
                repetido es alto o bajo**. Un corte de energía puede
                congelar la lectura en cualquier valor, incluido cero.

                **De noche esta prueba no se evalúa** (`NO_EVALUABLE`): un
                GHI=0 sostenido por horas ahí es normal, no una falla.

                ```yaml
                persistencia_n: 15  # Forstinger et al. (2021), "log-down"
                ```

                **Nota histórica:** la versión anterior de esta prueba
                exigía que el valor repetido superara un piso de magnitud
                (GHI > 50 W/m²) para evitar falsos positivos nocturnos —
                eso excluía justo el caso de un apagón diurno con GHI=0
                sostenido, delegando ese caso a "sensor caído". Pero esa
                prueba solo funciona con columnas TEMP/RH, que Udenar 2023
                (fuente derivada del inversor) no tiene, así que ese
                escenario quedaba sin cubrir. Ahora el corte día/noche usa
                la geometría solar en vez de un umbral de magnitud, así que
                persistencia sí cubre ceros sostenidos en pleno día.
                """)
            with col2:
                st.info("""
                **Resultado en Udenar (tras la corrección):**
                **0,185%** de flags de persistencia (194 registros).

                **Ejemplo real:** 27 de agosto de 2023, de 08:50 a 13:45h,
                GHI=0 sostenido con el sol entre 39° y 81° de elevación
                (cielo despejado teórico de hasta 1170 W/m²) — un apagón
                real detectado. Esas horas ya no cuentan como medición
                válida, y la serie reconstruida las rellena con una
                estimación coherente en vez de heredar el cero.
                """)

        with st.expander("📈 Prueba 10 — Simetría AM/PM"):
            col1, col2 = st.columns([1.1, 1])
            with col1:
                st.markdown("**Fórmula:**")
                st.latex(r"""
                Sim_{AM/PM} = \frac{|E_{AM} - E_{PM}|}{E_{AM} + E_{PM}}
                \quad \times 100\ \%
                """)
                st.markdown(r"""
                donde $E_{AM}$ y $E_{PM}$ son la energía acumulada (integral de GHI)
                antes y después del **mediodía solar** de cada día.

                **Criterio de flag:**
                $$Sim_{AM/PM} > 30\%$$

                En ausencia de obstrucciones y con nubosidad simétrica,
                la curva de irradiancia es aproximadamente simétrica
                alrededor del mediodía solar.

                Una asimetría > 30% puede indicar:
                - Obstrucción del horizonte en un lado
                - Sensor mal nivelado o con sombra parcial
                - Patrón climático real (nubosidad convectiva de tarde)

                ```yaml
                simetria_umbral_pct: 30.0
                ```
                """)
            with col2:
                st.markdown("**Hallazgo en Udenar:**")
                st.warning("""
                Asimetría AM/PM: **23.1%** de los registros
                (resolución nativa), y **11.6%** de los días
                evaluables en la re-verificación horaria de OE3.

                Es, junto con el GHI nocturno, el modo de falla
                más frecuente en Udenar — bastante más marcado
                que ERL, PPL o persistencia (todos cercanos a 0%).
                Recordemos que esta prueba es **diagnóstica**: no
                cuenta para el `% usable` (ver criterio de
                usabilidad más abajo), precisamente porque una
                asimetría AM/PM puede reflejar nubosidad real
                (patrón climático) y no necesariamente una falla
                de sensor.
                """)

        st.markdown("---")

        # ── CRITERIO DE USABILIDAD ────────────────────────────────────────────
        subseccion("Criterio de usabilidad — combinación de pruebas")

        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.markdown("**Regla base:**")
            st.latex(r"""
            \text{USABLE si } \nexists \text{ prueba con FLAG}
            """)
            st.markdown("""
            Un dato se declara **NO_USABLE** si al menos una prueba
            evaluable lo marcó FLAG. Las pruebas NO_EVALUABLE no cuentan
            ni a favor ni en contra.
            """)
        with col_u2:
            st.markdown("**Regla de islas:**")
            st.markdown("""
            Extensión opcional: un segmento continuo de datos OK con
            duración **< 60 minutos**, rodeado por tramos NO_USABLE,
            también se descarta.

            *Motivación:* unos pocos minutos de datos buenos rodeados
            de tramos malos no son útiles para análisis ni para el
            entrenamiento del modelo de site-adaptation.
            """)

        panel_info(
            "Todos los umbrales usados en estas pruebas están documentados "
            "y justificados individualmente en la configuración interna "
            "del proyecto."
        )


    # ── TAB 1: Resumen comparativo ────────────────────────────────────────────
    elif _sel_oe1 == _etq_oe1[1]:
        seccion("Resumen comparativo")

        # KPIs
        if df_usab is not None:
            cols = st.columns(max(len(df_usab), 1))
            for i, (_, row) in enumerate(df_usab.iterrows()):
                fuente = FUENTES_LABEL.get(row["fuente"], row["fuente"])
                pct = row["pct_usable_base"]
                pct_islas = row["pct_usable_islas"]
                # HTML propio en vez de st.metric(): el delta nativo de
                # Streamlit (delta_color="inverse") pinta un fondo
                # rosado/rojo que no se puede recolorear -- acá se fuerza
                # azul oscuro con letras blancas en negrilla.
                cols[i].markdown(
                    f"<div style='margin-bottom:0.75rem'>"
                    f"<div style='font-size:0.85rem; color:#555; margin-bottom:2px'>{fuente}</div>"
                    f"<div style='font-weight:700; font-size:1.9rem; color:#111; line-height:1.1'>{pct:.1f}%</div>"
                    f"<div style='display:inline-block; margin-top:6px; background:{C_TEAL}; "
                    f"color:#ffffff; font-weight:700; font-size:0.8rem; padding:3px 10px; "
                    f"border-radius:12px'>↑ con islas: {pct_islas:.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Tabla de resolución temporal por fuente: se prefiere a un histograma
        # de intervalos, poco informativo cuando la resolución es casi
        # constante.
        _crudos_resumen = cargar_crudos_global()
        if _crudos_resumen:
            subseccion("Resolución temporal por fuente")
            _filas_res = []
            for _n, _s in _crudos_resumen.items():
                _r = resumen_intervalo_serie(_s)
                _filas_res.append({
                    "Fuente": _n,
                    "Intervalo típico": f"{_r['intervalo_min']:.1f} min" if _r["intervalo_min"] else "—",
                    "% regular": f"{_r['pct_regular']:.1f}%" if _r["pct_regular"] is not None else "—",
                    "N registros": f"{_r['n_registros']:,}",
                    "Período": _r["periodo"],
                })
            tabla_html(pd.DataFrame(_filas_res))

        tooltip("""
        **% de datos utilizables** — proporción de datos que pasan TODAS las
        pruebas físicas y de experto aplicables a cada fuente.

        - **Regla base:** un dato se descarta si alguna prueba aplicable lo marca FLAG.
        - **Regla islas:** además, se descartan datos "isla" (pocos minutos de datos
          válidos rodeados por largos tramos de datos inválidos), ya que son de poca
          utilidad práctica.

        Solcast puede mostrar un % distinto al de la estación terrestre porque
        cada fuente tiene su propia cobertura y resolución nativa (Solcast es
        continuo cada 5 min, sin huecos; la estación terrestre puede tener
        gaps) — los datos nocturnos se marcan NO_EVALUABLE en las pruebas de
        día, lo que también afecta el denominador de cada una.
        """)

        # Columnas de fuente presentes en las tablas comparativas (4 en total:
        # 2 satelitales -- una por caso/año -- + 2 terrestres). Se resuelven
        # dinámicamente por si alguna fuente no llegó a procesarse todavía.
        _COLS_FUENTE_ORDEN = ["solcast", "udenar", "solcast_2025", "udenar_est"]
        _NOMBRES_FUENTE_SHOW = {
            "solcast": "Solcast 2023", "udenar": "Udenar (terrestre)",
            "solcast_2025": "Solcast 2025", "udenar_est": NOMBRE_UDENAR_EST,
        }

        # Tabla flags físicos
        if df_fisico is not None:
            subseccion("Pruebas físicas (BSRN) — % de datos con flag")
            _cols_disp = [c for c in _COLS_FUENTE_ORDEN if c in df_fisico.columns]
            df_show = df_fisico[["prueba"] + _cols_disp].copy()
            df_show.columns = ["Prueba"] + [_NOMBRES_FUENTE_SHOW[c] for c in _cols_disp]
            pruebas_map = {
                "flag_neg_GHI": "GHI negativo",
                "flag_noche_GHI": "GHI nocturno",
                "flag_ppl_GHI": "GHI sobre límite físico",
                "flag_erl_GHI": "GHI extremadamente raro",
                "flag_sensor_caido": "Sensor caído",
            }
            df_show["Prueba"] = df_show["Prueba"].map(lambda x: pruebas_map.get(x, x))
            df_show = df_show[df_show["Prueba"].isin(pruebas_map.values())]
            _cols_num_show = [_NOMBRES_FUENTE_SHOW[c] for c in _cols_disp]
            _fmt_pct = lambda x: f"{x:.2f}%" if isinstance(x, float) else x
            tabla_html(df_show, formato={c: _fmt_pct for c in _cols_num_show})
            boton_descarga_csv(df_show, "oe1_qc_fisico_comparativo.csv")

        # Tabla flags experto
        if df_experto is not None:
            subseccion("Pruebas de experto — % de datos con flag")
            _cols_disp_e = [c for c in _COLS_FUENTE_ORDEN if c in df_experto.columns]
            df_exp = df_experto[["prueba"] + _cols_disp_e].copy()
            df_exp.columns = ["Prueba"] + [_NOMBRES_FUENTE_SHOW[c] for c in _cols_disp_e]
            pruebas_exp = {
                "flag_cierre": "Cierre radiativo",
                "flag_csi": "Índice cielo claro (CSI)",
                "flag_simetria_ampm": "Simetría AM/PM",
                "flag_ktests": "K-tests (Kt, K, Kn)",
                "flag_roc": "Tasa de cambio",
                "flag_persistencia": "Persistencia",
            }
            df_exp["Prueba"] = df_exp["Prueba"].map(lambda x: pruebas_exp.get(x, x))
            df_exp = df_exp[df_exp["Prueba"].isin(pruebas_exp.values())]
            _cols_num_exp = [_NOMBRES_FUENTE_SHOW[c] for c in _cols_disp_e]
            _fmt_pct_e = lambda x: f"{x:.2f}%" if isinstance(x, float) else x
            tabla_html(df_exp, formato={c: _fmt_pct_e for c in _cols_num_exp})
            boton_descarga_csv(df_exp, "oe1_qc_experto_comparativo.csv")

        # Evolución de la depuración por etapa (crudo -> QC físico -> QC experto)
        subseccion("Evolución de la depuración de datos por etapa")
        panel_info(
            "Para la fuente seleccionada se muestra la serie de GHI en tres capas superpuestas: "
            "todos los datos crudos, los que pasan el control de calidad físico, y los que además "
            "pasan el control de calidad de experto (K-tests, cierre radiativo, CSI, persistencia, simetría, etc.). "
            "Cada etapa es acumulativa sobre la anterior."
        )
        _claves_evol = {
            "Udenar (terrestre)": "udenar", "Solcast": "solcast",
            NOMBRE_UDENAR_EST: CLAVE_UDENAR_EST, "Solcast 2025": "solcast_2025",
        }
        _fuente_evol = st.selectbox(
            "Fuente", list(_claves_evol.keys()), index=0, key="sel_fuente_evolucion_qc",
        )
        _clave_evol = _claves_evol[_fuente_evol]
        _df_evol = cargar_procesado(f"{_clave_evol}_qc_completo.csv")
        _fig_evol = fig_evolucion_qc_plotly(_df_evol, _fuente_evol, COLORES_FUENTES.get(_fuente_evol, C_BLUE))
        if _fig_evol is not None:
            st.plotly_chart(_fig_evol, use_container_width=True, key=f"evolucion_qc_{_clave_evol}")
        else:
            st.info(f"No hay datos procesados disponibles para {_fuente_evol}.")

    # ── TAB 2: Fuente satelital ───────────────────────────────────────────────
    elif _sel_oe1 == _etq_oe1[2]:
        seccion("Solcast — Fuente satelital", C_BLUE)
        st.caption("Cada caso del proyecto usa su propia serie satelital de referencia "
                   "(mismo sitio, distinto año) — elige cuál ver abajo.")

        _VARIANTES_SOLCAST = [
            ("Solcast 2023", "Solcast", "solcast", "2023"),
            ("Solcast 2025", "Solcast 2025", "solcast_2025", "2025"),
        ]
        _sel_sol = tabs_perezosas([f"📡 {n}" for n, _, _, _ in _VARIANTES_SOLCAST], key="tabs_sol_sel")

        for _i_sol, (_nombre_sol, _clave_crudo_sol, _tag_proc_sol, _sufijo_sol) in enumerate(_VARIANTES_SOLCAST):
            if _sel_sol == f"📡 {_nombre_sol}":
                _color_sol = COLORES_FUENTES.get(_clave_crudo_sol, C_BLUE)

                def _fallback_sol(nombre_png, caption, _sufijo=_sufijo_sol, _nombre=_nombre_sol):
                    """Muestra la figura estática nombre_png con su caption si existe
                    para el caso; si no, avisa que faltan los datos procesados."""
                    if _sufijo == "2023":
                        mostrar_figura(nombre_png, caption)
                        boton_descarga_fig(nombre_png)
                    else:
                        st.info(f"Aún no hay datos procesados para {_nombre}. "
                                "Corre verificar_pieza1_solar.py → …pieza4_usabilidad.py.")

                panel_info(f"{_nombre_sol} es el producto satelital de referencia "
                           f"(pixel Udenar). Resolución: 5 min · {_sufijo_sol}.")

                _sol_s = cargar_crudos_global().get(_clave_crudo_sol)

                if _sol_s is not None:
                    st.plotly_chart(
                        fig_cobertura_mensual_plotly(_sol_s, _nombre_sol, _color_sol),
                        use_container_width=True, key=f"cob_mensual_solcast_{_sufijo_sol}",
                    )
                    panel_info(
                        "Cuántos registros hay por mes frente a lo esperado a la resolución nativa "
                        "de la fuente — barras completas indican meses sin huecos de cobertura."
                    )
                    _res_sol = resumen_intervalo_serie(_sol_s)
                    _cm1, _cm2, _cm3 = st.columns(3)
                    _cm1.metric("Intervalo típico", f"{_res_sol['intervalo_min']:.1f} min" if _res_sol["intervalo_min"] else "—")
                    _cm2.metric("N registros", f"{_res_sol['n_registros']:,}")
                    _cm3.metric("Período", _res_sol["periodo"])
                else:
                    _fallback_sol("00_solcast_cobertura_mensual.png", f"Cobertura mensual de {_nombre_sol}")

                subseccion(f"Serie temporal GHI — {_nombre_sol} (datos crudos completos, con ceros)")
                if _sol_s is not None:
                    _ss = _sol_s.dropna()
                    _fig_sol = go.Figure(go.Scattergl(
                        x=_ss.index, y=_ss.values, mode="lines",
                        line=dict(color=_color_sol, width=0.8),
                        fill="tozeroy", fillcolor="rgba(0,0,0,0.06)", name=_nombre_sol,
                        hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.1f} W/m²<extra></extra>",
                    ))
                    _fig_sol.update_layout(
                        height=340, plot_bgcolor="white", paper_bgcolor="white",
                        showlegend=False, margin=dict(t=10, b=20, l=60, r=20),
                        xaxis=dict(
                            showgrid=False, tickformat="%b %Y",
                            rangeselector=dict(buttons=[
                                dict(count=7,  label="1S",  step="day",   stepmode="backward"),
                                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                                dict(step="all", label="Todo"),
                            ], bgcolor="#f0f6f9", activecolor="#1F4E5F", font=dict(size=10)),
                            rangeslider=dict(visible=True, thickness=0.07),
                        ),
                        yaxis=dict(title=dict(text="GHI [W/m²]", font=dict(size=11)),
                                   gridcolor="#eeeeee", rangemode="tozero"),
                    )
                    st.plotly_chart(_fig_sol, use_container_width=True, key=f"sol_ghi_serie_{_sufijo_sol}")
                    panel_info(
                        "GHI crudo sin ningún filtro, resolución nativa. El ciclo día/noche y la "
                        "variabilidad por nubosidad son visibles a simple vista; usa el selector "
                        "de rango para acercarte a un periodo específico."
                    )
                else:
                    _fallback_sol("01_serie_ghi_solcast.png", f"Serie diaria de GHI — {_nombre_sol}")

                _df_solcast_geo = cargar_procesado(f"{_tag_proc_sol}_qc_completo.csv")

                _fig_hm_ghi = fig_heatmap_diario_plotly(_df_solcast_geo, "GHI", _nombre_sol)
                if _fig_hm_ghi is not None:
                    st.plotly_chart(_fig_hm_ghi, use_container_width=True, key=f"heatmap_ghi_solcast_{_sufijo_sol}")
                else:
                    _fallback_sol("04_heatmap_ghi_solcast.png", "Patrón anual de GHI (día × hora)")
                _fig_hm_dni = fig_heatmap_diario_plotly(_df_solcast_geo, "DNI", _nombre_sol)
                if _fig_hm_dni is not None:
                    st.plotly_chart(_fig_hm_dni, use_container_width=True, key=f"heatmap_dni_solcast_{_sufijo_sol}")
                else:
                    _fallback_sol("24_heatmap_dni_solcast.png", "Patrón anual de DNI (día × hora)")
                panel_info(
                    "Cada columna es un día del año y cada fila una hora del día — el color muestra "
                    "el valor medio de irradiancia en ese momento. La franja diurna (horas centrales, "
                    "colores más cálidos al mediodía) debe verse consistente todo el año; huecos o "
                    "bandas verticales en blanco indican periodos sin datos."
                )

                _fig_csi_sol = fig_csi_anual_plotly(_df_solcast_geo, _nombre_sol, _color_sol)
                if _fig_csi_sol is not None:
                    st.plotly_chart(_fig_csi_sol, use_container_width=True, key=f"csi_anual_solcast_{_sufijo_sol}")
                    panel_info(
                        "Índice de cielo claro (CSI = GHI medido / GHI de cielo despejado teórico) a lo "
                        "largo del año — valores cercanos a 1 indican cielo despejado, valores más bajos "
                        "indican nubosidad. Sirve para detectar sesgos sostenidos del sensor (CSI "
                        "consistentemente muy alto o muy bajo, no explicado por el clima)."
                    )
                else:
                    _fallback_sol("14_solcast_csi_anual.png", "Índice de cielo claro (CSI) a lo largo del año")

                subseccion(f"Diagnósticos exclusivos de {_nombre_sol} (3 componentes)")
                if _df_solcast_geo is not None and {"GHI", "DNI", "DHI", "zenith"}.issubset(_df_solcast_geo.columns):
                    _dfc = _df_solcast_geo[_df_solcast_geo["GHI"] > 50].copy()
                    _dfc["GHI_calc"] = _dfc["DNI"] * np.cos(np.radians(_dfc["zenith"])) + _dfc["DHI"]
                    _flag_col = "flag_cierre" if "flag_cierre" in _dfc.columns else None
                    _fig_cierre = go.Figure()
                    if _flag_col:
                        for _et, _color in [("OK", "#22AA44"), ("FLAG", "#E63946")]:
                            _sub = _dfc[_dfc[_flag_col] == _et]
                            if len(_sub) == 0:
                                continue
                            _fig_cierre.add_trace(go.Scatter(
                                x=_sub["GHI_calc"], y=_sub["GHI"], mode="markers",
                                marker=dict(size=3, color=_color, opacity=0.5),
                                name=f"{_et} ({len(_sub):,})",
                                hovertemplate="Calculado: %{x:.0f} W/m²<br>Medido: %{y:.0f} W/m²<extra></extra>",
                            ))
                    else:
                        _fig_cierre.add_trace(go.Scatter(
                            x=_dfc["GHI_calc"], y=_dfc["GHI"], mode="markers",
                            marker=dict(size=3, color=_color_sol, opacity=0.5),
                            name=_nombre_sol,
                        ))
                    _lim = float(max(_dfc["GHI_calc"].max(), _dfc["GHI"].max())) * 1.02
                    _fig_cierre.add_trace(go.Scatter(
                        x=[0, _lim], y=[0, _lim], mode="lines",
                        line=dict(color="black", dash="dash", width=1.2),
                        name="Cierre perfecto (1:1)",
                    ))
                    _fig_cierre.update_layout(
                        title="Cierre radiativo — GHI medido vs calculado (DNI·cos(SZA) + DHI)",
                        height=460, plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="DNI·cos(SZA) + DHI [W/m²]", gridcolor="#f0f0f0"),
                        yaxis=dict(title="GHI medido [W/m²]", gridcolor="#f0f0f0"),
                        legend=dict(orientation="h", y=1.06, x=0),
                        margin=dict(t=60, b=20, l=60, r=20),
                    )
                    st.plotly_chart(_fig_cierre, use_container_width=True, key=f"cierre_radiativo_solcast_{_sufijo_sol}")
                    panel_info(
                        "Verifica la consistencia física entre las 3 componentes de irradiancia: "
                        "GHI medido debería igualar DNI·cos(SZA) + DHI (línea diagonal 1:1). Puntos "
                        "rojos (FLAG) se alejan de esa diagonal más de la tolerancia permitida."
                    )
                else:
                    _fallback_sol("13_solcast_cierre_radiativo.png",
                                   "Cierre radiativo: GHI medido vs GHI calculado (DNI·cos(SZA) + DHI)")

                _ktests_figs = fig_ktests_plotly(_df_solcast_geo)
                col5, col6 = st.columns(2)
                with col5:
                    if _ktests_figs is not None:
                        st.plotly_chart(_ktests_figs[0], use_container_width=True, key=f"ktests_k_vs_kt_{_sufijo_sol}")
                    else:
                        _fallback_sol("21_ktests_solcast_k_vs_kt.png", "K vs Kt — límites BSRN")
                with col6:
                    if _ktests_figs is not None:
                        st.plotly_chart(_ktests_figs[1], use_container_width=True, key=f"ktests_kn_vs_kt_{_sufijo_sol}")
                    else:
                        _fallback_sol("21_ktests_solcast_kn_vs_kt.png", "Kn vs Kt — límites BSRN")

                tooltip(f"""
                **K-tests (índices de claridad, estándar BSRN)**

                - **Kt = GHI / (ETN · cos(SZA)):** índice de transparencia atmosférica global.
                - **K = DHI / GHI:** fracción difusa.
                - **Kn = DNI / ETN:** índice de irradiancia directa normalizada.

                Los tres deben mantenerse dentro de los límites físicos de la atmósfera terrestre.
                Un dato fuera de esos límites (rojo) indica un error en la medición o en el modelo.
                {_nombre_sol} tiene **0% de flags** en estas pruebas (esperado en un producto satelital).
                """)

                _fig_sesgo_sol = fig_sesgo_dni_azimut_plotly(_df_solcast_geo, _nombre_sol, _color_sol)
                if _fig_sesgo_sol is not None:
                    st.plotly_chart(_fig_sesgo_sol, use_container_width=True, key=f"sesgo_dni_azimut_solcast_{_sufijo_sol}")
                else:
                    _fallback_sol("20_sesgo_dni_azimut_solcast.png",
                                   "Sesgo de DNI (medido − calculado por cierre) vs azimut solar")

                tooltip(f"""
                **Sesgo DNI vs azimut** — Si hubiera una obstrucción del horizonte en una
                dirección específica, veríamos un patrón de sesgo negativo concentrado en ese
                azimut. La mediana (línea negra) se mantiene cerca de 0 en todo el rango:
                no hay evidencia de obstrucción direccional en {_nombre_sol}.
                """)

                subseccion("Envolvente solar — mapa elevación × azimut")
                _geo_solcast_env = _df_solcast_geo
                _fig_env_ghi_sol = fig_envolvente_plotly(_geo_solcast_env, "GHI", _nombre_sol)
                if _fig_env_ghi_sol is not None:
                    st.plotly_chart(_fig_env_ghi_sol, use_container_width=True, key=f"envolvente_ghi_solcast_{_sufijo_sol}")
                else:
                    _fallback_sol("22_envolvente_ghi_solcast.png",
                                   f"{_nombre_sol} — envolvente máxima de GHI. Arco continuo, sin anomalías.")
                panel_info(
                    "Para cada celda del cielo (elevación × azimut solar) se pinta el valor máximo de "
                    "GHI registrado en todo el periodo. Una celda que nunca alcanza el valor esperado "
                    "para esa elevación (un \"valle oscuro\" que no es el borde del arco) sugiere "
                    "obstrucción del horizonte en esa dirección; un arco fragmentado en vez de continuo "
                    "indica huecos de cobertura temporal. La forma de \"dos arcos\" es normal para Pasto: "
                    "al estar casi sobre el ecuador (1.21°N), el sol pasa al norte en junio y al sur en "
                    "diciembre, dando dos trayectorias distintas."
                )

                _fig_env_dni_sol = fig_envolvente_plotly(_geo_solcast_env, "DNI", _nombre_sol)
                if _fig_env_dni_sol is not None:
                    st.plotly_chart(_fig_env_dni_sol, use_container_width=True, key=f"envolvente_dni_solcast_{_sufijo_sol}")
                else:
                    _fallback_sol("23_envolvente_dni_solcast.png",
                                   f"{_nombre_sol} — envolvente máxima de DNI. "
                                   "La componente directa es más sensible a la nubosidad.")
                panel_info(
                    "Misma lógica que la envolvente de GHI, pero para la componente directa (DNI) — "
                    "más sensible a la nubosidad, por lo que su envolvente máxima es más irregular "
                    "incluso sin ningún problema del sensor."
                )

    # ── TAB 3: Estación terrestre — subtabs por caso (mismo sitio, 2 años) ────
    elif _sel_oe1 == _etq_oe1[3]:
        seccion("Udenar — Estación terrestre", C_RED)
        st.caption("Mismo sitio, dos periodos con datasets distintos — elige el año a explorar.")
        _sel_udenar_oe1 = tabs_perezosas(["📅 2023", "📅 2025"], key="tabs_udenar_oe1_sel")

        _render_udenar_oe1(_sel_udenar_oe1)

    # ── TAB 4: Métricas cruzadas ──────────────────────────────────────────────
    elif _sel_oe1 == _etq_oe1[4]:
        seccion("Métricas cruzadas — GHI Solcast vs estación terrestre")

        tooltip(
            "Qué mide esta tabla:\n\n"
            "Compara el GHI de cada estación terrestre contra Solcast (la referencia), "
            "en 2 versiones: **Crudo** (todos los datos, sin ningún filtro) y "
            "**Post-QC** (solo las horas que pasan el control de calidad propio de esa "
            "estación) — así se ve directamente el efecto de quedarse solo con datos "
            "confiables. Métricas sobre promedios horarios diurnos (Solcast > 10 W/m²).\n\n"
            "- MBE positivo = Solcast sobreestima la estación.\n"
            "- Si Post-QC mejora frente a Crudo (MAE/RMSE más bajos, Pearson r/R² más "
            "altos), es evidencia de que el control de calidad retiene los datos más "
            "confiables — si no mejora, algo anda mal en las pruebas de QC."
        )

        _estaciones_mc = [("udenar", "Udenar (terrestre)"), (CLAVE_UDENAR_EST, NOMBRE_UDENAR_EST)]
        _filas_mc = []
        for _clave_mc, _nombre_mc in _estaciones_mc:
            _res_mc = calcular_metricas_qc_vs_solcast(_clave_mc)
            if _res_mc is None:
                continue
            for _etq_mc, _label_mc in [("crudo", "Crudo"), ("post_qc", "Post-QC")]:
                _m = _res_mc.get(_etq_mc)
                if _m is None or pd.isna(_m.get("MAE")):
                    continue
                _filas_mc.append({
                    "Estacion": _nombre_mc, "Versión": _label_mc,
                    "N horas": _m["n_horas"], "Periodo": _m["periodo"],
                    "MBE [W/m2]": round(_m["MBE"], 1), "rMBE [%]": round(_m["rMBE"], 1),
                    "MAE [W/m2]": round(_m["MAE"], 1), "RMSE [W/m2]": round(_m["RMSE"], 1),
                    "rRMSE [%]": round(_m["rRMSE"], 1),
                    "Pearson r": round(_m["Pearson_R"], 3), "R2": round(_m["R2"], 3),
                })

        if not _filas_mc:
            st.warning("No hay datos suficientes para calcular las métricas cruzadas.")
        else:
            df_show = pd.DataFrame(_filas_mc)

            subseccion("Tabla de métricas por estación — crudo vs. post-control de calidad")
            _cols_num = ["MAE [W/m2]", "RMSE [W/m2]", "rRMSE [%]"]
            _cols_mbe = ["MBE [W/m2]", "rMBE [%]"]
            _cols_r   = ["Pearson r", "R2"]
            _fmt = {"N horas": "{:,.0f}"}
            for _c in _cols_mbe:
                _fmt[_c] = "{:+.1f}"
            for _c in _cols_num:
                _fmt[_c] = "{:.1f}"
            for _c in _cols_r:
                _fmt[_c] = "{:.3f}"
            tabla_html(df_show, formato=_fmt)
            boton_descarga_csv(df_show, "oe1_metricas_cruzadas_qc.csv",
                               "Descargar tabla (.csv)")

            subseccion("Comparación visual — RMSE crudo vs. post-QC, por estación")
            _dfc_crudo = df_show[df_show["Versión"] == "Crudo"]
            _dfc_qc = df_show[df_show["Versión"] == "Post-QC"]
            _fb = go.Figure()
            _fb.add_trace(go.Bar(name="RMSE — Crudo", x=_dfc_crudo["Estacion"],
                y=_dfc_crudo["RMSE [W/m2]"], marker_color="#cccccc", opacity=0.9,
                text=[f"{v:.1f}" for v in _dfc_crudo["RMSE [W/m2]"]], textposition="outside"))
            _fb.add_trace(go.Bar(name="RMSE — Post-QC", x=_dfc_qc["Estacion"],
                y=_dfc_qc["RMSE [W/m2]"], marker_color=C_TEAL, opacity=0.9,
                text=[f"{v:.1f}" for v in _dfc_qc["RMSE [W/m2]"]], textposition="outside"))
            _fb.update_layout(barmode="group", height=380,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Arial", size=11),
                legend=dict(orientation="h", y=1.05, x=0),
                margin=dict(t=40, b=40, l=40, r=20),
                yaxis=dict(title="RMSE [W/m²]", gridcolor="#e5e5e5"))
            st.plotly_chart(_fb, use_container_width=True, key="metricas_cruzadas_rmse_qc")
            panel_info(
                "Es normal que 'Post-QC' cambie poco frente a 'Crudo' (unos pocos W/m², a "
                "veces incluso con el mismo orden de magnitud): el control de calidad descarta "
                "solo una fracción pequeña de las horas (típicamente 1-3% en estas estaciones) "
                "y esas horas son las que tienen errores de sensor evidentes "
                "(valores imposibles, picos, saturación) — no está diseñado para corregir el "
                "desacople sistemático entre un sensor puntual en tierra y un pixel satelital "
                "bajo nubosidad variable, que es la causa principal de la diferencia frente a "
                "Solcast. Por eso tampoco hay que esperar que el sesgo (MBE) mejore siempre "
                "tras el QC: como se filtran horas puntuales en ambas direcciones, el sesgo "
                "puede moverse un poco para cualquier lado sin que eso indique un problema. "
                "Los valores en el orden de 60-150 W/m² son normales para GHI horario "
                "satelital vs. estación terrestre en un sitio de nubosidad convectiva como "
                "Pasto — no es un error de la app. La reducción grande de error viene después, "
                "en la etapa de site-adaptation, que sí corrige ese desacople sistemático "
                "(ver la pestaña correspondiente)."
            )

    elif _sel_oe1 == _etq_oe1[5]:
        render_oe1_diagrama_flujo()

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PÁGINA: OE2  ██████████████████
# ─────────────────────────────────────────────────────────────────────────────

elif "OE2" in pagina:
    st.markdown(
        f"""<div style='background:linear-gradient(135deg,{C_BLUE},{C_TEAL});
        color:white; padding:28px 36px; border-radius:16px; margin-bottom:1.5rem'>
        <div style='font-size:2.25rem; font-weight:600; color:white; margin:0'>📐 OE2 — Site-Adaptation</div>
        <p style='color:#d0eaf4; margin-top:6px; font-size:1rem'>
        Corrección y reconstrucción de la estación terrestre usando Solcast
        como referencia · 9 modelos × 2 variantes · 2 casos (Udenar 2023 y
        {NOMBRE_UDENAR_EST})</p>
        </div>""",
        unsafe_allow_html=True,
    )

    tooltip("""
    **¿Qué se hizo en esta etapa de site-adaptation? (pivote metodológico)**

    Originalmente se corregía Solcast (satelital) usando la estación
    terrestre como referencia — la dirección clásica de *site-adaptation*
    en la literatura. Tras revisar con el asesor el volumen de datos
    inválidos/faltantes de la estación terrestre (ver la etapa de control
    de calidad), se decidió invertir la lógica: **Solcast, que es
    continuo y consistente, se usa ahora como referencia para corregir y
    rellenar huecos en la estación terrestre.**

    **Por estación:**
    1. **Emparejamiento a resolución nativa (5 min)** Solcast↔estación,
       filtrando **instante por instante** por la bandera de usabilidad
       (no por una ventana de fechas fija).
    2. **División temporal honesta (70/30 sin fuga)** sobre el conjunto
       supervisado (los instantes con medición confiable).
    3. **9 modelos** comparados; se elige el de menor RMSE entre los
       físicamente válidos (mismo criterio que ya se valida en la
       verificación final).
    4. **2 variantes de reconstrucción** con el modelo ganador, aplicado a
       TODO el período disponible: *total* (reemplaza toda la serie) y
       *selectiva* (conserva la medición cuando ya era válida, corrige solo
       lo flageado y rellena huecos).
    """)

    _etq_oe2 = [
        "📐 Marco metodológico",
        "🔬 Udenar",
        "📊 Resumen — mejoras por etapa",
        "🧭 Diagrama de flujo",
    ]
    _sel_oe2 = tabs_perezosas(_etq_oe2, key="tabs_oe2_sel")

    if _sel_oe2 == _etq_oe2[0]:
        render_oe2_marco_metodologico()
    elif _sel_oe2 == _etq_oe2[1]:
        seccion("Udenar — Site-adaptation", C_RED)
        st.caption("Mismo sitio, dos periodos con datasets distintos — elige el año a explorar.")
        _sel_udenar_oe2 = tabs_perezosas(["📅 2023", "📅 2025"], key="tabs_udenar_oe2_sel")
        if _sel_udenar_oe2 == "📅 2023":
            render_tab_reconstruccion("udenar", "Udenar (terrestre)", C_RED)
        elif _sel_udenar_oe2 == "📅 2025":
            render_tab_reconstruccion(CLAVE_UDENAR_EST, NOMBRE_UDENAR_EST, C_UDENAR_EST)

    elif _sel_oe2 == _etq_oe2[2]:
        render_oe2_resumen_mejoras()

    elif _sel_oe2 == _etq_oe2[3]:
        render_oe2_diagrama_flujo()


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PÁGINA: OE3  ██████████████████
# ─────────────────────────────────────────────────────────────────────────────
elif "OE3" in pagina:
    st.markdown(
        f"""<div style='background:linear-gradient(135deg,{C_GOLD}cc,{C_TEAL});
        color:white; padding:28px 36px; border-radius:16px; margin-bottom:1.5rem'>
        <div style='font-size:2.25rem; font-weight:600; color:white; margin:0'>📊 OE3 — Desempeño y Re-QC</div>
        <p style='color:#f0f8e8; margin-top:6px; font-size:1rem'>
        Re-control de calidad de la estación reconstruida · crudo vs. corregido
        (total / selectivo) · 2 casos (Udenar 2023 y {NOMBRE_UDENAR_EST})</p>
        </div>""",
        unsafe_allow_html=True,
    )

    tooltip("""
    **¿Qué se hizo en esta etapa de verificación de desempeño?**

    Esta etapa responde la pregunta que la corrección por sí sola no puede contestar:
    **¿el dato reconstruido, además de ser más preciso, sigue siendo físicamente aceptable?**

    Para responderla, se re-aplica exactamente el mismo control de calidad de la
    etapa inicial al dato crudo de cada estación terrestre Y a sus 2 versiones
    corregidas (total / selectiva), sobre todo el período de aplicación. Una
    mejora real requiere:
    - Menor error estadístico (MAE, RMSE ↓; R² ↑) del modelo ganador vs. el baseline
      (Solcast sin corregir), evaluado en el conjunto de prueba honesto.
    - Mayor porcentaje de datos físicamente utilizables tras la reconstrucción.
    - Reducción real de banderas en las pruebas que capturan huecos y errores de
      sensor (nocturno, límites físicos, persistencia) — no solo en promedio.
    """)

    _etq_oe3 = [
        "📐 Marco metodológico",
        "📉 Evolución completa",
        "🖼️ Mosaico de diagnóstico",
        "🔬 Udenar",
        "🧭 Diagrama de flujo",
    ]
    _sel_oe3 = tabs_perezosas(_etq_oe3, key="tabs_oe3_sel")

    if _sel_oe3 == _etq_oe3[0]:
        render_oe3_marco_metodologico()
    elif _sel_oe3 == _etq_oe3[1]:
        # El resumen comparativo (% usable por etapa, modelo ganador, RMSE
        # por etapa) se renderiza solo dentro de esta pestaña. Fuera de los
        # bloques de pestaña, Streamlit lo mostraría en todas ellas.
        render_oe3_evolucion_completa()
        st.markdown("<hr>", unsafe_allow_html=True)
        render_oe3_resumen_4estaciones()
    elif _sel_oe3 == _etq_oe3[2]:
        render_oe3_mosaico_diagnostico()
    elif _sel_oe3 == _etq_oe3[3]:
        seccion("Udenar — Desempeño y Re-QC", C_RED)
        st.caption("Mismo sitio, dos periodos con datasets distintos — elige el año a explorar.")
        _sel_udenar_oe3 = tabs_perezosas(["📅 2023", "📅 2025"], key="tabs_udenar_oe3_sel")
        if _sel_udenar_oe3 == "📅 2023":
            render_tab_oe3_estacion("udenar", "Udenar (terrestre)", C_RED)
        elif _sel_udenar_oe3 == "📅 2025":
            render_tab_oe3_estacion(CLAVE_UDENAR_EST, NOMBRE_UDENAR_EST, C_UDENAR_EST)

    elif _sel_oe3 == _etq_oe3[4]:
        render_oe3_diagrama_flujo()

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PÁGINA: NUEVO DATASET (demo de generalización)  ████████
# ─────────────────────────────────────────────────────────────────────────────
elif "Nuevo dataset" in pagina:
    st.markdown(
        f"""<div style='background:linear-gradient(135deg,{C_TEAL},{C_BLUE});
        color:white; padding:28px 36px; border-radius:16px; margin-bottom:1.5rem'>
        <div style='font-size:2.25rem; font-weight:600; color:white; margin:0'>🌍 Nuevo dataset — demo de generalización</div>
        <p style='color:#d0eaf4; margin-top:6px; font-size:1rem'>
        Corre el pipeline completo (control de calidad → site-adaptation → verificación)
        EN VIVO sobre datos de cualquier ciudad, no solo Solcast + Udenar</p></div>""",
        unsafe_allow_html=True,
    )

    tooltip("""
    **¿Qué hace esta pestaña?**

    Todo lo demás en esta app muestra resultados ya precalculados de los 2
    casos fijos de este proyecto (Udenar 2023 y Udenar estación 2025, cada
    uno con su propio Solcast pareja). Esta pestaña es
    distinta: sube tú un CSV satelital (ej. NASA POWER) y un CSV terrestre
    (ej. una estación IDEAM) de **cualquier ciudad** — por ejemplo Bogotá —
    y la app corre, en vivo, exactamente el mismo procesamiento que generó
    todos los resultados de Udenar: control de calidad físico/experto,
    entrenamiento y selección del
    mejor de los 9 modelos de reconstrucción, y re-verificación de la serie
    corregida. Sirve para demostrar que la metodología generaliza a otro
    sitio, no solo a los datos con los que se construyó.

    **Requisitos mínimos de cada archivo:** una columna de fecha/hora y una
    columna de GHI (irradiancia global horizontal, en W/m²). El satelital
    puede además traer DNI/DHI (si las trae, se activan cierre radiativo y
    K-tests, igual que con Solcast); si no, solo se aplican las pruebas que
    no requieren las 3 componentes (CSI, tasa de cambio, persistencia,
    simetría AM/PM).
    """)

    if not QCSOLAR_DISPONIBLE:
        st.error(
            f"No se pudo cargar el motor de procesamiento (`src/qcsolar`): "
            f"**{QCSOLAR_ERROR}**"
        )
        st.markdown("""
        Esta pestaña necesita paquetes que la app de solo-lectura no
        instaló originalmente (`requirements_app.txt` solo traía
        streamlit/pandas/plotly/pillow). Con tu entorno virtual activado:

        ```
        pip install pvlib scikit-learn pyyaml
        ```

        y vuelve a refrescar la página.
        """)
    else:
        if "nd_resultado" not in st.session_state:
            st.session_state["nd_resultado"] = None

        subseccion("1. Parámetros del sitio nuevo")
        st.caption("Valores precargados para Bogotá — cámbialos para cualquier otra ciudad.")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        nd_lat = col_s1.number_input("Latitud [°]", value=4.7110, format="%.4f", key="nd_lat")
        nd_lon = col_s2.number_input("Longitud [°]", value=-74.0721, format="%.4f", key="nd_lon")
        nd_alt = col_s3.number_input("Altitud [m]", value=2640.0, step=10.0, key="nd_alt")
        nd_tz = col_s4.selectbox(
            "Zona horaria (IANA)",
            ["America/Bogota", "America/Lima", "America/Mexico_City", "America/Santiago", "UTC"],
            index=0, key="nd_tz",
        )
        nd_nombre_raw = st.text_input(
            "Nombre corto del sitio/estación (usado en nombres de columna internos)",
            value="bogota_ideam", key="nd_nombre",
        )
        nd_nombre = "".join(c if c.isalnum() else "_" for c in nd_nombre_raw.strip().lower()) or "nuevo_sitio"

        st.markdown("---")
        subseccion("2. Subir los dos datasets")
        col_u1, col_u2 = st.columns(2)

        df_sat_std, df_terr_std = None, None

        with col_u1:
            st.markdown("**📡 Dataset satelital** (ej. NASA POWER)")
            archivo_sat = st.file_uploader("CSV satelital", type=["csv"], key="nd_sat_file")
            if archivo_sat is not None:
                df_sat_raw = _leer_csv_generico(archivo_sat)
                if df_sat_raw is None or df_sat_raw.empty:
                    st.error("No se pudo leer el CSV satelital (revisa el separador/formato).")
                else:
                    st.caption(f"Vista previa ({len(df_sat_raw):,} filas):")
                    tabla_html(df_sat_raw.head(5))
                    cols_sat = list(df_sat_raw.columns)
                    cs1, cs2 = st.columns(2)
                    col_fecha_sat = cs1.selectbox("Columna fecha/hora", cols_sat, key="nd_sat_fecha")
                    col_ghi_sat = cs2.selectbox("Columna GHI [W/m²]", cols_sat, key="nd_sat_ghi")
                    cs3, cs4 = st.columns(2)
                    col_dni_sat = cs3.selectbox("Columna DNI (opcional)", ["— no disponible —"] + cols_sat, key="nd_sat_dni")
                    col_dhi_sat = cs4.selectbox("Columna DHI (opcional)", ["— no disponible —"] + cols_sat, key="nd_sat_dhi")
                    ya_tz_sat = st.checkbox(
                        "La columna de fecha ya incluye desplazamiento de zona horaria",
                        key="nd_sat_ya_tz",
                    )
                    try:
                        fechas_sat, _fmt_sat = _parsear_fecha_generica(df_sat_raw[col_fecha_sat])
                        if ya_tz_sat and fechas_sat.dt.tz is not None:
                            fechas_sat = fechas_sat.dt.tz_convert(nd_tz)
                        else:
                            fechas_sat = fechas_sat.dt.tz_localize(
                                nd_tz, nonexistent="shift_forward", ambiguous="NaT")
                        df_sat_std = pd.DataFrame({"datetime": fechas_sat})
                        df_sat_std["GHI"] = pd.to_numeric(df_sat_raw[col_ghi_sat], errors="coerce")
                        if col_dni_sat != "— no disponible —":
                            df_sat_std["DNI"] = pd.to_numeric(df_sat_raw[col_dni_sat], errors="coerce")
                        if col_dhi_sat != "— no disponible —":
                            df_sat_std["DHI"] = pd.to_numeric(df_sat_raw[col_dhi_sat], errors="coerce")
                        _frac_sana_sat = _fraccion_fechas_sanas(df_sat_std["datetime"])
                        df_sat_std = (df_sat_std.dropna(subset=["datetime"])
                                      .sort_values("datetime").drop_duplicates("datetime")
                                      .reset_index(drop=True))
                        if df_sat_std.empty:
                            st.error("Ninguna fecha se pudo interpretar — revisa la columna de fecha/hora.")
                            df_sat_std = None
                        elif _frac_sana_sat < 0.5:
                            st.error(
                                f"⚠️ La columna `{col_fecha_sat}` no parece ser una fecha válida "
                                f"(la mayoría de los valores caen fuera de 1990–hoy, ej. "
                                f"`{df_sat_std['datetime'].iloc[0]}`). Es probable que hayas elegido "
                                f"la columna equivocada, o que el formato de fecha no se haya "
                                f"reconocido. Si tu columna viene como `AAAAMMDDHH` pegado "
                                f"(típico de NASA POWER), verifica que solo tenga dígitos."
                            )
                            df_sat_std = None
                        else:
                            st.success(
                                f"✅ {len(df_sat_std):,} registros válidos — "
                                f"{df_sat_std['datetime'].min()} a {df_sat_std['datetime'].max()}"
                                + (f" (formato de fecha detectado: `{_fmt_sat}`)" if _fmt_sat else "")
                            )
                    except Exception as _e_norm_sat:
                        st.error(f"Error al normalizar el dataset satelital: {_e_norm_sat}")
                        df_sat_std = None

        with col_u2:
            st.markdown("**🏔️ Dataset terrestre** (ej. estación IDEAM)")
            archivo_terr = st.file_uploader("CSV terrestre", type=["csv"], key="nd_terr_file")
            if archivo_terr is not None:
                df_terr_raw = _leer_csv_generico(archivo_terr)
                if df_terr_raw is None or df_terr_raw.empty:
                    st.error("No se pudo leer el CSV terrestre (revisa el separador/formato).")
                else:
                    st.caption(f"Vista previa ({len(df_terr_raw):,} filas):")
                    tabla_html(df_terr_raw.head(5))
                    cols_terr = list(df_terr_raw.columns)
                    ct1, ct2 = st.columns(2)
                    col_fecha_terr = ct1.selectbox("Columna fecha/hora", cols_terr, key="nd_terr_fecha")
                    col_ghi_terr = ct2.selectbox("Columna GHI [W/m²]", cols_terr, key="nd_terr_ghi")
                    ya_tz_terr = st.checkbox(
                        "La columna de fecha ya incluye desplazamiento de zona horaria",
                        key="nd_terr_ya_tz",
                    )
                    try:
                        fechas_terr, _fmt_terr = _parsear_fecha_generica(df_terr_raw[col_fecha_terr])
                        if ya_tz_terr and fechas_terr.dt.tz is not None:
                            fechas_terr = fechas_terr.dt.tz_convert(nd_tz)
                        else:
                            fechas_terr = fechas_terr.dt.tz_localize(
                                nd_tz, nonexistent="shift_forward", ambiguous="NaT")
                        df_terr_std = pd.DataFrame({"datetime": fechas_terr})
                        df_terr_std["GHI"] = pd.to_numeric(df_terr_raw[col_ghi_terr], errors="coerce")
                        _frac_sana_terr = _fraccion_fechas_sanas(df_terr_std["datetime"])
                        df_terr_std = (df_terr_std.dropna(subset=["datetime"])
                                       .sort_values("datetime").drop_duplicates("datetime")
                                       .reset_index(drop=True))
                        if df_terr_std.empty:
                            st.error("Ninguna fecha se pudo interpretar — revisa la columna de fecha/hora.")
                            df_terr_std = None
                        elif _frac_sana_terr < 0.5:
                            st.error(
                                f"⚠️ La columna `{col_fecha_terr}` no parece ser una fecha válida "
                                f"(la mayoría de los valores caen fuera de 1990–hoy, ej. "
                                f"`{df_terr_std['datetime'].iloc[0]}`). Es probable que hayas elegido "
                                f"la columna equivocada — en exportaciones tipo IDEAM/datos abiertos, "
                                f"la fecha real suele estar en una columna aparte (ej. `Fecha`, "
                                f"`FechaObservacion`), no en el código de la estación. Desplázate a la "
                                f"derecha en la vista previa para verla."
                            )
                            df_terr_std = None
                        else:
                            st.success(
                                f"✅ {len(df_terr_std):,} registros válidos — "
                                f"{df_terr_std['datetime'].min()} a {df_terr_std['datetime'].max()}"
                                + (f" (formato de fecha detectado: `{_fmt_terr}`)" if _fmt_terr else "")
                            )
                    except Exception as _e_norm_terr:
                        st.error(f"Error al normalizar el dataset terrestre: {_e_norm_terr}")
                        df_terr_std = None

        nd_periodo_inicio, nd_periodo_fin = None, None
        _datasets_ok = (df_sat_std is not None and df_terr_std is not None
                        and len(df_sat_std) > 0 and len(df_terr_std) > 0)
        if _datasets_ok:
            st.markdown("---")
            subseccion("2.5. Periodo de análisis")
            _ov_inicio = max(df_sat_std["datetime"].min(), df_terr_std["datetime"].min())
            _ov_fin = min(df_sat_std["datetime"].max(), df_terr_std["datetime"].max())
            if pd.isna(_ov_inicio) or pd.isna(_ov_fin) or _ov_inicio >= _ov_fin:
                st.error(
                    "Los dos datasets no tienen ningún periodo de fechas en común — "
                    "revisa el mapeo de columnas de fecha/hora antes de continuar."
                )
            else:
                st.caption(
                    f"Datos disponibles en **ambas** fuentes: **{_ov_inicio.date()}** a "
                    f"**{_ov_fin.date()}**. Por defecto se usa todo ese traslape para "
                    f"entrenar/evaluar los modelos; acótalo si quieres (ej. un solo año), "
                    f"con los selectores de abajo. La reconstrucción final (site-adaptation) siempre "
                    f"cubre todo el rango disponible de la fuente satelital, sin importar "
                    f"lo que elijas aquí — esto solo afecta con qué datos se entrena/evalúa."
                )
                cp1, cp2 = st.columns(2)
                _fecha_ini_sel = cp1.date_input(
                    "Fecha inicial", value=_ov_inicio.date(),
                    min_value=_ov_inicio.date(), max_value=_ov_fin.date(), key="nd_periodo_ini",
                )
                _fecha_fin_sel = cp2.date_input(
                    "Fecha final", value=_ov_fin.date(),
                    min_value=_ov_inicio.date(), max_value=_ov_fin.date(), key="nd_periodo_fin",
                )
                if _fecha_ini_sel > _fecha_fin_sel:
                    st.error("La fecha inicial no puede ser posterior a la fecha final.")
                else:
                    nd_periodo_inicio = str(_fecha_ini_sel)
                    nd_periodo_fin = str(_fecha_fin_sel)
                    _dias_sel = (_fecha_fin_sel - _fecha_ini_sel).days + 1
                    st.caption(f"Periodo seleccionado para entrenar/evaluar: {_dias_sel:,} días.")

        st.markdown("---")
        subseccion("3. Ejecutar")
        listo = (_datasets_ok and nd_periodo_inicio is not None and nd_periodo_fin is not None)
        if not listo:
            st.info("Sube y mapea correctamente ambos datasets, y elige un periodo válido, "
                     "para habilitar el botón.")

        if st.button("▶️ Ejecutar pipeline completo (control de calidad → site-adaptation → verificación)",
                      disabled=not listo, type="primary", key="nd_run"):
            with st.spinner("Corriendo QC físico/experto, emparejando, entrenando 7 "
                            "modelos, eligiendo ganador y re-verificando..."):
                try:
                    st.session_state["nd_resultado"] = ejecutar_pipeline_generico(
                        df_sat_std, df_terr_std, nd_lat, nd_lon, nd_alt, nd_tz,
                        nombre_objetivo=nd_nombre,
                        periodo_inicio=nd_periodo_inicio, periodo_fin=nd_periodo_fin,
                    )
                    st.session_state["nd_nombre_display"] = nd_nombre_raw.strip() or "Sitio nuevo"
                except Exception as _e_pipeline:
                    st.session_state["nd_resultado"] = {
                        "error": f"Error inesperado corriendo el pipeline: "
                                 f"{type(_e_pipeline).__name__}: {_e_pipeline}"
                    }

        nd_resultado = st.session_state.get("nd_resultado")
        if nd_resultado is not None:
            if nd_resultado.get("error"):
                st.error(nd_resultado["error"])
            else:
                st.markdown("---")
                _nombre_disp = st.session_state.get("nd_nombre_display", "Sitio nuevo")
                subseccion(f"Resultados — {_nombre_disp}")

                _resumen = nd_resultado["resumen_emparejamiento"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Muestras supervisadas (train+test)", f"{_resumen['muestras_emparejadas_supervisado']:,.0f}")
                c2.metric("Muestras de aplicación (todo el periodo)", f"{_resumen['muestras_aplicacion_total']:,.0f}")
                c3.metric("Sesgo satelital−terrestre",
                          f"{_resumen['sesgo_medio_antes (Solcast-estacion, W/m2)']:+.1f} W/m²")
                c4.metric("Correlación cruda", f"{_resumen['correlacion_antes']:.3f}")
                _periodo_ini_disp = nd_resultado.get("periodo_inicio")
                _periodo_fin_disp = nd_resultado.get("periodo_fin")
                _prefijo_periodo = (
                    f"Periodo de entrenamiento/evaluación: {_periodo_ini_disp} a {_periodo_fin_disp} · "
                    if _periodo_ini_disp and _periodo_fin_disp
                    else "⚠️ Resultado de una ejecución anterior a esta versión (sin filtro de "
                         "periodo) — dale otra vez a \"Ejecutar pipeline completo\" para aplicar el "
                         "periodo seleccionado arriba · "
                )
                panel_info(
                    f"{_prefijo_periodo}"
                    f"Train: {_resumen['n_train']:,} horas · "
                    f"Test: {_resumen['n_test']:,} horas (split temporal 70/30, nunca aleatorio) · "
                    f"Resolución detectada del satelital: "
                    f"~{nd_resultado['muestras_h_sat']:.0f} muestras/hora · "
                    f"Fuente satelital tratada como "
                    f"{'3 componentes (GHI+DNI+DHI)' if nd_resultado['fuente_sat']=='solcast' else 'solo GHI'}."
                )

                subseccion("Ranking de modelos (conjunto de prueba, nunca visto en entrenamiento)")
                _tm = nd_resultado["tabla_metricas_enriquecida"].sort_values("RMSE", ascending=False)
                _ganador = nd_resultado["ganador"]
                _colores_rk = [
                    C_TEAL if m == _ganador else "#cccccc" if not fv else "#a8d5e2"
                    for m, fv in zip(_tm["modelo"], _tm["fisicamente_valido"])
                ]
                _fig_rk = go.Figure(go.Bar(
                    x=_tm["RMSE"], y=_tm["modelo"], orientation="h",
                    marker_color=_colores_rk,
                    text=[f"{v:.1f}" for v in _tm["RMSE"]], textposition="outside",
                    hovertemplate="%{y}<br>RMSE: %{x:.2f} W/m²<extra></extra>",
                ))
                _fig_rk.update_layout(
                    title=f"RMSE por modelo (test) — ganador: {_ganador}",
                    height=360, plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(t=50, b=20, l=10, r=40),
                    xaxis=dict(title="RMSE [W/m²]", gridcolor="#f0f0f0"), yaxis=dict(title=""),
                )
                st.plotly_chart(_fig_rk, use_container_width=True, key="nd_ranking")
                panel_info(
                    "Cada barra es un modelo, ordenado por RMSE en el conjunto de prueba (menor "
                    "es mejor). Color destacado = modelo ganador; gris = descartado (no "
                    "físicamente válido); celeste = válido pero no ganador."
                )

                _tabla_show = nd_resultado["tabla_metricas"][
                    ["ranking", "modelo", "MAE", "RMSE", "MBE", "rRMSE", "Pearson_R", "R2"]
                ].copy()
                _tabla_show.columns = ["#", "Modelo", "MAE [W/m²]", "RMSE [W/m²]", "MBE [W/m²]",
                                        "rRMSE [%]", "Pearson r", "R²"]
                tabla_html(_tabla_show)
                boton_descarga_csv(_tabla_show, f"nuevo_dataset_{nd_nombre}_metricas.csv")

                subseccion("Evolución: crudo vs. corregido")
                _df_serie = nd_resultado["df_corregido"]
                _fig_serie = go.Figure()
                _fig_serie.add_trace(go.Scatter(
                    x=_df_serie["datetime"], y=_df_serie[f"GHI_{nd_nombre}_crudo"], mode="lines",
                    line=dict(color="#B02A2A", width=1.1), opacity=0.75,
                    name="Crudo (medido)",
                    hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
                ))
                _fig_serie.add_trace(go.Scatter(
                    x=_df_serie["datetime"], y=_df_serie["GHI_corregido_total"], mode="lines",
                    line=dict(color=COLOR_CORREGIDO_TOTAL, width=1.6), opacity=0.9,
                    name="Corregido — total",
                    hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
                ))
                _fig_serie.add_trace(go.Scatter(
                    x=_df_serie["datetime"], y=_df_serie["GHI_corregido_selectivo"], mode="lines",
                    line=dict(color=COLOR_CORREGIDO_SELECTIVO, width=1.2), opacity=0.9,
                    name="Corregido — selectivo",
                    hovertemplate="%{x|%d %b %Y %H:%M}<br>GHI: %{y:.0f} W/m²<extra></extra>",
                ))
                _fig_serie.update_layout(
                    title=f"{_nombre_disp} — GHI: crudo vs. corregido (total / selectivo)",
                    height=380, plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(t=40, b=20, l=60, r=170),
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                                font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
                                bordercolor="#ccc", borderwidth=1, itemsizing="constant"),
                    xaxis=dict(showgrid=False, tickformat="%b %Y",
                               rangeslider=dict(visible=True, thickness=0.07)),
                    yaxis=dict(title=dict(text="GHI [W/m²]", font=dict(size=11)),
                               gridcolor="#eeeeee", rangemode="tozero"),
                )
                st.plotly_chart(_fig_serie, use_container_width=True, key="nd_evolucion")
                panel_info(
                    "Serie temporal completa del sitio nuevo: dato satelital crudo contra la "
                    "serie ya corregida por el modelo ganador. Usa el selector de rango para "
                    "acercarte a un periodo específico."
                )

                subseccion("Dispersión — satelital crudo vs. corregido (post verificación final)")
                _test = nd_resultado["test"]
                _col_obj = f"GHI_{nd_nombre}"
                _pred_ganador = nd_resultado["predicciones_test"][_ganador]
                _lim_nd = float(max(
                    _test["GHI_solcast"].max(), _test[_col_obj].max(), _pred_ganador.max()
                )) * 1.05
                _fig_disp_nd = make_subplots(rows=1, cols=2, subplot_titles=(
                    f"Satelital crudo vs {_nombre_disp}", f"{_ganador} vs {_nombre_disp}"))
                _fig_disp_nd.add_trace(go.Scatter(
                    x=_test[_col_obj], y=_test["GHI_solcast"], mode="markers",
                    marker=dict(color="#000000", size=4, opacity=0.5),
                    name="Satelital crudo", showlegend=False,
                ), row=1, col=1)
                _fig_disp_nd.add_trace(go.Scatter(
                    x=[0, _lim_nd], y=[0, _lim_nd], mode="lines",
                    line=dict(color="black", dash="dash", width=1), showlegend=False,
                ), row=1, col=1)
                _fig_disp_nd.add_trace(go.Scatter(
                    x=_test[_col_obj], y=_pred_ganador, mode="markers",
                    marker=dict(color=C_TEAL, size=4, opacity=0.5),
                    name=_ganador, showlegend=False,
                ), row=1, col=2)
                _fig_disp_nd.add_trace(go.Scatter(
                    x=[0, _lim_nd], y=[0, _lim_nd], mode="lines",
                    line=dict(color="black", dash="dash", width=1), showlegend=False,
                ), row=1, col=2)
                _fig_disp_nd.update_xaxes(title_text=f"GHI {_nombre_disp} [W/m²]", gridcolor="#f0f0f0")
                _fig_disp_nd.update_yaxes(title_text="GHI predicho [W/m²]", gridcolor="#f0f0f0", col=1)
                _fig_disp_nd.update_layout(
                    height=440, plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(t=40, b=20, l=60, r=20),
                )
                st.plotly_chart(_fig_disp_nd, use_container_width=True, key="nd_dispersion")
                panel_info(
                    "Cada punto es una hora del conjunto de prueba: eje X = GHI medido en el "
                    "sitio, eje Y = predicción. Izquierda = satelital crudo (sin corregir); "
                    "derecha = modelo ganador. Entre más pegados estén los puntos a la diagonal "
                    "punteada (1:1), mejor la predicción."
                )

                subseccion("% de datos físicamente utilizables (re-control de calidad final) — crudo vs. corregido")
                col_a, col_b, col_c = st.columns(3)
                for _col, _label, _pct, _c in [
                    (col_a, "Crudo (sin corregir)", nd_resultado["pct_usable_crudo"], "#4d4d4d"),
                    (col_b, "Corregido — total", nd_resultado["pct_usable_total"], COLOR_CORREGIDO_TOTAL),
                    (col_c, "Corregido — selectivo", nd_resultado["pct_usable_selectivo"], COLOR_CORREGIDO_SELECTIVO),
                ]:
                    _col.markdown(
                        f"<div style='background:{C_LIGHT}; border-radius:12px; padding:18px; "
                        f"border-left:5px solid {_c}'>"
                        f"<b style='font-size:0.95rem; color:{_c}'>{_label}</b><br>"
                        f"<span style='font-size:2.2rem; font-weight:700; color:{_c}'>{_pct:.2f}%</span>"
                        f"</div>", unsafe_allow_html=True
                    )

                panel_info(
                    "Este resultado se generó en vivo, en tu computador, corriendo exactamente "
                    "el mismo procesamiento (control de calidad → site-adaptation → verificación) "
                    "que produjo todos los resultados de Udenar — cambia únicamente el sitio "
                    "(coordenadas) y los datos de entrada. Sirve como evidencia de que la "
                    "metodología no está atada a las 2 fuentes originales."
                )

                st.download_button(
                    "⬇ Descargar serie corregida completa (.csv)",
                    nd_resultado["df_corregido"].to_csv(index=False).encode("utf-8"),
                    file_name=f"nuevo_dataset_{nd_nombre}_ghi_corregido.csv", mime="text/csv",
                )

