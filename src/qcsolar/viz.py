"""
viz.py
------
Generacion de figuras del paquete. Cada funcion dibuja UN tipo de figura
y la guarda en outputs/figuras/ con resolucion apta para publicacion.

La logica de cada figura se define una sola vez en este modulo y se
reutiliza desde los scripts de verificacion, los documentos (a traves de
los PNG guardados) y la aplicacion web (Streamlit invoca estas mismas
funciones), de modo que no se duplica codigo de graficacion.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: solo guarda archivos
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .report import DIR_FIGURAS
from .config import CONFIG

# --- Estilo visual comun para todas las figuras ---
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
})

# Paleta de colores, en orden: fuente satelital, estacion terrestre,
# cielo despejado, datos marcados (FLAG) y modelo elegido.
COLOR_SOLCAST = "#2b6f8f"
COLOR_ALVERNIA = "#c1666b"
COLOR_CLEAR = "#e9a544"
COLOR_FLAG = "#d1495b"
COLOR_ELEGIDO = "#2a9d8f"

def _guardar(fig, nombre: str) -> Path:
    """Guarda la figura en outputs/figuras/ con fondo blanco, la cierra y
    devuelve la ruta del archivo generado."""
    ruta = DIR_FIGURAS / nombre
    fig.savefig(ruta, facecolor="white")
    plt.close(fig)
    print(f"   [figura guardada]  figuras/{nombre}")
    return ruta


# ============================================================
#  Figuras de datos crudos / serie temporal
# ============================================================

def fig_serie_temporal(df: pd.DataFrame, col: str, titulo: str, nombre_archivo: str,
                       color: str = COLOR_SOLCAST, agregar: str = "D") -> Path:
    """
    Serie temporal de una variable agregada con la frecuencia `agregar`
    (por defecto diaria, 'D'), mostrando el promedio de los valores
    diurnos de cada periodo para no saturar la figura.
    """
    d = df[["datetime", col]].dropna().set_index("datetime")
    # Se filtran solo valores diurnos (> 0) antes de agregar, para evitar
    # que los ceros nocturnos reduzcan artificialmente el promedio diario.
    _col_num = pd.to_numeric(d[col], errors="coerce")
    serie = _col_num[_col_num > 0].resample(agregar).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(serie.index, serie.values, lw=1.0, color=color)
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(f"{col} [W/m2]" if col in ("GHI", "DNI", "DHI") else col)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    return _guardar(fig, nombre_archivo)


def fig_cobertura_temporal(df: pd.DataFrame, titulo: str, nombre_archivo: str,
                           color: str = COLOR_ALVERNIA) -> Path:
    """
    Grafico de cobertura: numero de registros por dia. Revela los huecos
    de la serie, especialmente util en estaciones terrestres con
    interrupciones de registro.
    """
    conteo = df.set_index("datetime").resample("D").size()
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(conteo.index, conteo.values, width=1.0, color=color)
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("registros/dia")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    return _guardar(fig, nombre_archivo)


# ============================================================
#  Figuras de geometria solar
# ============================================================

def fig_heatmap_diario(df: pd.DataFrame, col: str, titulo: str, nombre_archivo: str,
                       cmap: str = "inferno") -> Path:
    """
    Mapa de calor dia del anio (eje X) vs hora del dia (eje Y), al estilo
    de las herramientas de QC experto. Muestra de un vistazo el patron
    anual de la irradiancia (percentil 99 como limite de color).
    """
    d = df[["datetime", col]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()
    d["doy"] = d["datetime"].dt.dayofyear
    d["hora"] = d["datetime"].dt.hour + d["datetime"].dt.minute / 60
    pivote = d.pivot_table(index="hora", columns="doy", values=col, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12, 5))
    vmax = np.nanpercentile(d[col], 99) if len(d) else 1
    im = ax.imshow(pivote.to_numpy(), origin="lower", aspect="auto", cmap=cmap,
                   vmin=0, vmax=vmax,
                   extent=[1, 366, 0, 24])
    ax.set_title(titulo)
    ax.set_xlabel("Dia del anio")
    ax.set_ylabel("Hora local")
    ax.set_yticks(range(0, 25, 3))
    ax.grid(False)
    plt.colorbar(im, ax=ax, label=f"{col} [W/m2]")
    return _guardar(fig, nombre_archivo)


def fig_dia_vs_clearsky(df: pd.DataFrame, fecha: str, titulo: str, nombre_archivo: str) -> Path:
    """
    Curva de un dia concreto (`fecha`): GHI medido vs GHI de cielo
    despejado (clear-sky), si la columna 'ghi_clear' existe. Permite
    apreciar visualmente la nubosidad de ese dia.
    """
    d = df.copy()
    d["fecha"] = d["datetime"].dt.date
    dia = d[d["fecha"] == pd.to_datetime(fecha).date()].sort_values("datetime")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if "ghi_clear" in dia.columns:
        ax.plot(dia["datetime"], dia["ghi_clear"], lw=1.6, color=COLOR_CLEAR,
                label="Cielo despejado (modelo)")
    ax.plot(dia["datetime"], dia["GHI"], lw=1.4, color=COLOR_SOLCAST, label="GHI medido")
    ax.set_title(titulo)
    ax.set_xlabel("Hora")
    ax.set_ylabel("GHI [W/m2]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend()
    return _guardar(fig, nombre_archivo)


def fig_trayectoria_solar(df: pd.DataFrame, titulo: str, nombre_archivo: str) -> Path:
    """
    Dispersion azimut (X) vs elevacion (Y) coloreada por GHI: la 'huella'
    del sol a lo largo del anio sobre el sitio. Usa una muestra de hasta
    40 000 instantes diurnos para mantener la figura legible.
    """
    d = df[df["elevation"] > 0][["azimuth", "elevation", "GHI"]].dropna()
    muestra = d.sample(min(len(d), 40000), random_state=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(muestra["azimuth"], muestra["elevation"], c=muestra["GHI"],
                    s=4, cmap="inferno", alpha=0.6)
    ax.set_title(titulo)
    ax.set_xlabel("Azimut solar [grados]  (90=Este, 180=Sur, 270=Oeste)")
    ax.set_ylabel("Elevacion solar [grados]")
    plt.colorbar(sc, ax=ax, label="GHI [W/m2]")
    return _guardar(fig, nombre_archivo)


def fig_envolvente_solar(df: pd.DataFrame, col: str, titulo: str, nombre_archivo: str,
                          bins_azimut: float = 2.0, bins_elevacion: float = 1.0,
                          cmap: str = "turbo"):
    """
    Mapa elevacion x azimut coloreado por el VALOR MAXIMO de `col`
    observado en cada celda de la grilla durante todo el periodo, al
    estilo de la envolvente solar de la metodologia de QC experto.

    A diferencia de fig_trayectoria_solar (dispersion instante a
    instante), esta figura agrupa el cielo en celdas y toma el maximo por
    celda: el "techo" de irradiancia alcanzado en cada combinacion
    elevacion-azimut. Sirve para detectar obstrucciones del horizonte:
    una celda que nunca alcanza el maximo esperado para su elevacion (un
    "valle" oscuro que NO es el borde del horizonte) sugiere la sombra de
    un objeto fijo en esa direccion, y no ruido de muestreo. Las celdas
    sin datos quedan en blanco (no en 0) para no confundir "sin dato" con
    "irradiancia cero".

    Usa por defecto el colormap 'turbo' (alto contraste, sin grandes zonas
    oscuras) en lugar de colormaps oscuros como inferno/magma, para que
    las diferencias se distingan mejor a simple vista.
    Devuelve la ruta de la figura, o None si faltan columnas o datos.
    """
    req = {col, "elevation", "azimuth"}
    if not req.issubset(df.columns):
        return None
    d = df[[col, "elevation", "azimuth"]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[(d["elevation"] > 0) & d[col].notna()]
    if d.empty:
        return None

    az_edges = np.arange(0, 360 + bins_azimut, bins_azimut)
    el_edges = np.arange(0, 92, bins_elevacion)
    d["az_bin"] = pd.cut(d["azimuth"], az_edges, labels=False, include_lowest=True)
    d["el_bin"] = pd.cut(d["elevation"], el_edges, labels=False, include_lowest=True)

    tabla = d.groupby(["el_bin", "az_bin"])[col].max().reset_index()
    grid = np.full((len(el_edges) - 1, len(az_edges) - 1), np.nan)
    grid[tabla["el_bin"].astype(int), tabla["az_bin"].astype(int)] = tabla[col].to_numpy()
    grid_m = np.ma.masked_invalid(grid)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid_m, origin="lower", aspect="auto", cmap=cmap_obj,
                   extent=[0, 360, 0, 91], vmin=0)
    ax.set_title(titulo)
    ax.set_xlabel("Azimut solar [grados]  (90=Este, 180=Sur, 270=Oeste)")
    ax.set_ylabel("Elevacion solar [grados]")
    plt.colorbar(im, ax=ax, label=f"max {col} [W/m2]")
    return _guardar(fig, nombre_archivo)


# ============================================================
#  Figuras de COMPARACION entre las dos fuentes
# ============================================================

def _alinear_diario(solcast, alvernia, col="GHI"):
    """Promedia a valor diario (solo valores diurnos) la fuente satelital y
    la terrestre, y las une en una tabla con los dias comunes a ambas."""
    _s = solcast[["datetime", col]].dropna().set_index("datetime")[col]
    _a = alvernia[["datetime", col]].dropna().set_index("datetime")[col]
    # Se filtran solo valores diurnos (> 0) antes de promediar a diario.
    s = pd.to_numeric(_s, errors="coerce").pipe(lambda x: x[x > 0]).resample("D").mean()
    a = pd.to_numeric(_a, errors="coerce").pipe(lambda x: x[x > 0]).resample("D").mean()
    comp = pd.DataFrame({"Solcast": s, "Alvernia": a}).dropna()
    return comp


def fig_comparacion_series(solcast, alvernia, col="GHI", titulo=None, nombre_archivo=None):
    """
    Serie temporal con las dos fuentes (satelital y terrestre)
    superpuestas, en promedio diario. Muestra de un vistazo donde
    coinciden y donde difieren.
    """
    titulo = titulo or f"Comparacion Solcast vs Alvernia - {col} diario"
    nombre_archivo = nombre_archivo or f"comparacion_series_{col.lower()}.png"
    comp = _alinear_diario(solcast, alvernia, col)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(comp.index, comp["Solcast"], lw=1.1, color=COLOR_SOLCAST, label="Solcast (satelital)")
    ax.plot(comp.index, comp["Alvernia"], lw=1.1, color=COLOR_ALVERNIA, label="Alvernia (estacion)")
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(f"{col} [W/m2]" if col in ("GHI", "DNI", "DHI") else col)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend()
    return _guardar(fig, nombre_archivo)


def fig_dispersion_fuentes(solcast, alvernia, col="GHI", titulo=None, nombre_archivo=None):
    """
    Dispersion fuente satelital (X) vs estacion terrestre (Y) a valor
    diario, con la linea 1:1 y un recuadro con correlacion, sesgo medio y
    numero de dias. Si los puntos caen sobre la diagonal, las fuentes
    concuerdan; las desviaciones sistematicas son el sesgo que el
    site-adaptation debe corregir.
    """
    titulo = titulo or f"Solcast vs Alvernia - {col} (cada punto = un dia)"
    nombre_archivo = nombre_archivo or f"dispersion_fuentes_{col.lower()}.png"
    comp = _alinear_diario(solcast, alvernia, col)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(comp["Solcast"], comp["Alvernia"], s=14, alpha=0.5, color=COLOR_SOLCAST)
    lim = float(max(comp["Solcast"].max(), comp["Alvernia"].max())) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="Linea 1:1 (acuerdo perfecto)")
    # Correlacion y sesgo medio anotados como texto
    r = comp["Solcast"].corr(comp["Alvernia"])
    sesgo = (comp["Alvernia"] - comp["Solcast"]).mean()
    ax.text(0.04, 0.96,
            f"Correlacion r = {r:.3f}\nSesgo medio (Alv-Sol) = {sesgo:.1f} W/m2\nN dias = {len(comp)}",
            transform=ax.transAxes, va="top",
            bbox={"fc": "white", "ec": "0.7", "alpha": 0.9})
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(titulo)
    ax.set_xlabel(f"Solcast {col} [W/m2]")
    ax.set_ylabel(f"Alvernia {col} [W/m2]")
    ax.legend(loc="lower right")
    return _guardar(fig, nombre_archivo)


# ============================================================
#  Figuras ANTES/DESPUES (crudo vs normalizado) y panel de fuentes
# ============================================================

def fig_crudo_vs_normalizado(df_crudo, col_crudo, df_norm, col_norm,
                             nombre_fuente, nombre_archivo):
    """
    Dos paneles lado a lado: la misma variable en datos CRUDOS (izquierda)
    y NORMALIZADOS (derecha), ambos en promedio diario. Muestra que la
    normalizacion organiza la informacion (fechas, nombres, orden) SIN
    alterar los valores fisicos.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)

    # Panel crudo: usa la fecha auxiliar '_fecha' y agrega a diario.
    d0 = df_crudo[["_fecha", col_crudo]].dropna().set_index("_fecha")
    _v0 = pd.to_numeric(d0[col_crudo], errors="coerce")
    s0 = _v0[_v0 > 0].resample("D").mean()
    ax0.plot(s0.index, s0.values, lw=0.9, color="#9aa3ad")
    ax0.set_title(f"{nombre_fuente} - CRUDO ('{col_crudo}')")
    ax0.set_xlabel("Fecha")
    ax0.set_ylabel("Irradiancia [W/m2]")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Panel normalizado: usa la columna estandar 'datetime'.
    d1 = df_norm[["datetime", col_norm]].dropna().set_index("datetime")
    _v1 = pd.to_numeric(d1[col_norm], errors="coerce")
    s1 = _v1[_v1 > 0].resample("D").mean()
    ax1.plot(s1.index, s1.values, lw=0.9, color=COLOR_SOLCAST)
    ax1.set_title(f"{nombre_fuente} - NORMALIZADO ('{col_norm}')")
    ax1.set_xlabel("Fecha")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.suptitle(f"{nombre_fuente}: antes vs despues de normalizar (valores se conservan)",
                 y=1.02)
    return _guardar(fig, nombre_archivo)


def fig_panel_fuentes_crudas(solcast_crudo, alvernia_crudo, nombre_archivo):
    """
    Panel de 2x1 con el GHI crudo diario de cada fuente (satelital arriba,
    terrestre abajo) en el mismo eje temporal. Representa el punto de
    partida visual: los datos tal como se reciben, sin procesamiento.
    """
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Fuente satelital cruda (columna original 'ghi'). Se usa la MEDIA diaria
    # (no la mediana) porque en una serie de 5 min la mediana diaria es 0:
    # la mitad del dia es noche.
    d0 = solcast_crudo[["_fecha", "ghi"]].dropna().set_index("_fecha")
    _ghi0 = pd.to_numeric(d0["ghi"], errors="coerce")
    s0 = _ghi0[_ghi0 > 0].resample("D").mean()
    ax0.plot(s0.index, s0.values, lw=1.0, color=COLOR_SOLCAST)
    ax0.set_title("Solcast (satelital) - GHI crudo diario")
    ax0.set_ylabel("GHI [W/m2]")

    # Estacion terrestre cruda (columna original 'irradiance'). Media diaria
    # por consistencia con el panel superior.
    d1 = alvernia_crudo[["_fecha", "irradiance"]].dropna().set_index("_fecha")
    _irr1 = pd.to_numeric(d1["irradiance"], errors="coerce")
    s1 = _irr1[_irr1 > 0].resample("D").mean()
    ax1.plot(s1.index, s1.values, lw=1.0, color=COLOR_ALVERNIA)
    ax1.set_title("Alvernia (estacion) - irradiance crudo diario")
    ax1.set_ylabel("GHI [W/m2]")
    ax1.set_xlabel("Fecha")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.suptitle("Punto de partida: datos crudos de ambas fuentes", y=0.98)
    return _guardar(fig, nombre_archivo)


# ============================================================
#  Figuras del control de calidad fisico
# ============================================================

def fig_datos_flagueados(df, col, col_flag, titulo, nombre_archivo,
                         fuente_color=COLOR_SOLCAST):
    """
    Serie de irradiancia con los puntos marcados (FLAG) en `col_flag`
    resaltados en rojo. Muestra DONDE se ubican en el tiempo los datos
    sospechosos.
    """
    d = df[["datetime", col, col_flag]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=[col])
    ok = d[d[col_flag] == "OK"]
    flag = d[d[col_flag] == "FLAG"]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.scatter(ok["datetime"], ok[col], s=2, color=fuente_color, alpha=0.3, label="OK")
    ax.scatter(flag["datetime"], flag[col], s=6, color=COLOR_FLAG, alpha=0.8, label="FLAG")
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel(f"{col} [W/m2]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(markerscale=3)
    return _guardar(fig, nombre_archivo)


def fig_flags_por_mes(df, col_flag, titulo, nombre_archivo):
    """
    Barras con la cantidad de registros FLAG por mes. Revela si los
    problemas se concentran en alguna epoca (p. ej., un periodo de
    degradacion o suciedad del sensor).

    Si TODOS los meses dan 0 (una estacion o prueba sin ninguna violacion
    en todo el periodo, como puede ocurrir en la prueba de limite fisico
    PPL), el grafico vacio parece un error de renderizado: matplotlib
    autoescala el eje Y a un rango minusculo del tipo [-0.04, 0.04]. Para
    dejar claro que es un resultado real, se fija el eje Y y se anota el
    caso en la figura.
    """
    d = df[["datetime", col_flag]].copy()
    d["mes"] = d["datetime"].dt.tz_localize(None).dt.to_period("M").astype(str)
    conteo = d[d[col_flag] == "FLAG"].groupby("mes").size()
    todos_meses = d["mes"].drop_duplicates().sort_values()
    conteo = conteo.reindex(todos_meses, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(conteo)), conteo.values, color=COLOR_FLAG, alpha=0.8)
    ax.set_xticks(range(len(conteo)))
    ax.set_xticklabels(conteo.index, rotation=45, ha="right")
    ax.set_title(titulo)
    ax.set_xlabel("Mes")
    ax.set_ylabel("registros marcados (FLAG)")
    if conteo.max() == 0:
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "0 registros marcados en TODOS los meses\n(resultado real, no un error del grafico)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#2e7d32", fontweight="bold")
    return _guardar(fig, nombre_archivo)


# ============================================================
#  Figuras del QC experto
# ============================================================

def fig_cierre_radiativo(df, titulo, nombre_archivo):
    """
    Dispersion GHI medido vs GHI calculado (DNI*cos(SZA)+DHI) para GHI > 50
    W/m2. Los puntos sobre la diagonal cumplen el cierre radiativo; los
    alejados, no. Solo aplica a fuentes con las tres componentes
    (tipicamente la satelital); devuelve None si faltan columnas.
    """
    req = {"GHI", "DNI", "DHI", "zenith"}
    if not req.issubset(df.columns):
        return None
    cos = np.clip(np.cos(np.radians(df["zenith"])), 0, None)
    ghi = pd.to_numeric(df["GHI"], errors="coerce")
    ghi_calc = pd.to_numeric(df["DNI"], errors="coerce")*cos + pd.to_numeric(df["DHI"], errors="coerce")
    m = (ghi > 50) & ghi_calc.notna()
    g, gc = ghi[m], ghi_calc[m]
    idx = np.random.RandomState(1).choice(len(g), min(len(g), 30000), replace=False)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(gc.iloc[idx], g.iloc[idx], s=4, alpha=0.3, color=COLOR_SOLCAST)
    lim = float(max(g.max(), gc.max()))*1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="Cierre perfecto (1:1)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(titulo)
    ax.set_xlabel("DNI*cos(SZA) + DHI  [W/m2]")
    ax.set_ylabel("GHI medido [W/m2]")
    ax.legend(loc="lower right")
    return _guardar(fig, nombre_archivo)


def fig_csi_anual(df, titulo, nombre_archivo, color=COLOR_SOLCAST):
    """
    Indice de cielo claro (CSI = GHI/GHI_clear) mediano diario a lo largo
    del anio. Revela tendencias de calibracion: una caida sostenida del
    CSI puede indicar suciedad o degradacion del sensor.
    Devuelve None si faltan las columnas 'GHI' o 'ghi_clear'.
    """
    if "GHI" not in df.columns or "ghi_clear" not in df.columns:
        return None
    d = df[["datetime", "GHI", "ghi_clear"]].copy()
    d["GHI"] = pd.to_numeric(d["GHI"], errors="coerce")
    d = d[(d["GHI"] > 50) & (d["ghi_clear"] > 50)]
    d["csi"] = d["GHI"] / d["ghi_clear"]
    serie = d.set_index("datetime")["csi"].resample("D").median()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(serie.index, serie.values, lw=1.0, color=color)
    ax.axhline(1.0, color="green", ls="--", lw=1, alpha=0.7, label="CSI = 1 (ideal cielo despejado)")
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("CSI mediano diario")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend()
    return _guardar(fig, nombre_archivo)


# ============================================================
#  Figuras de diagnostico del QC experto
#  Inspiradas en la metodologia de QC experto de referencia y aplicadas
#  a las series del sitio; no reproducen figuras publicadas, sino que
#  construyen diagnosticos equivalentes sobre los datos propios.
# ============================================================

def fig_simetria_ampm_azimut(df, titulo, nombre_archivo, color=COLOR_SOLCAST,
                              min_muestras_por_mitad=10):
    """
    Visualiza la prueba de simetria AM/PM (test_simetria_ampm, expert.py)
    como dispersion: para cada dia, la asimetria relativa de energia AM vs
    PM (misma formula que expert.py: 100*(E_am-E_pm)/(E_am+E_pm)) contra el
    azimut solar de mediodia (azimut en el instante de mayor elevacion).
    Revela si la asimetria tiene una direccion preferente (compatible con
    obstruccion del horizonte o mala nivelacion del sensor) o si es solo
    ruido dia a dia. Aplica a cualquier fuente con GHI y geometria solar.

    FILTRO: un dia solo entra al grafico si AMBAS mitades (AM y PM) tienen
    al menos `min_muestras_por_mitad` lecturas validas de GHI y energia
    mayor que 0. Sin este filtro, una mitad del dia ausente (n=0 o todo en
    0) produce una asimetria de exactamente +-100% que no es un fenomeno
    radiativo sino un hueco de datos. El filtro aplica SOLO a esta figura
    de diagnostico; no modifica la prueba test_simetria_ampm de expert.py
    ni sus porcentajes de marcado.
    """
    req = {"GHI", "elevation", "azimuth"}
    if not req.issubset(df.columns):
        return None

    d = df[["datetime", "GHI", "elevation", "azimuth"]].copy()
    d["GHI"] = pd.to_numeric(d["GHI"], errors="coerce")
    d["fecha"] = d["datetime"].dt.date
    d["am"] = d["datetime"].dt.hour < 12
    d["dia"] = d["elevation"] > 0
    dia_df = d[d["dia"]]

    filas = []
    dias_excluidos = 0
    dias_totales = 0
    for fecha, grupo in dia_df.groupby("fecha"):
        dias_totales += 1
        ghi_am = grupo.loc[grupo["am"], "GHI"]
        ghi_pm = grupo.loc[~grupo["am"], "GHI"]
        n_am = ghi_am.notna().sum()
        n_pm = ghi_pm.notna().sum()
        am = ghi_am.sum()
        pm = ghi_pm.sum()
        if n_am < min_muestras_por_mitad or n_pm < min_muestras_por_mitad or am <= 0 or pm <= 0:
            dias_excluidos += 1
            continue
        asimetria_pct = 100 * (am - pm) / (am + pm)
        azimut_mediodia = grupo.loc[grupo["elevation"].idxmax(), "azimuth"]
        filas.append({"fecha": fecha, "asimetria_pct": asimetria_pct,
                       "azimut_mediodia": azimut_mediodia})

    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return None

    if dias_excluidos:
        print(f"   [simetria_azimut] {dias_excluidos} de {dias_totales} dias excluidos "
              f"(menos de {min_muestras_por_mitad} muestras o energia 0 en alguna mitad)")

    tol_pct = CONFIG["qc_experto"]["simetria_tol"] * 100

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(tabla["azimut_mediodia"], tabla["asimetria_pct"],
               s=12, alpha=0.55, color=color)
    ax.axhline(tol_pct, color=COLOR_FLAG, ls="--", lw=1,
               label=f"Umbral FLAG (+/-{tol_pct:.0f}%)")
    ax.axhline(-tol_pct, color=COLOR_FLAG, ls="--", lw=1)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(titulo)
    ax.set_xlabel("Azimut solar de mediodia [grados]  (0/360=Norte, 180=Sur)")
    ax.set_ylabel("Asimetria AM/PM [%]  =  100*(E_AM-E_PM)/(E_AM+E_PM)")
    ax.set_xlim(0, 360)
    ax.legend(fontsize=8)
    return _guardar(fig, nombre_archivo)


def fig_sesgo_dni_azimut(df, titulo, nombre_archivo, color=COLOR_SOLCAST):
    """
    Sesgo de DNI (medido menos calculado por cierre radiativo) vs azimut
    solar. DNI_calc se despeja de la MISMA ecuacion de cierre que usa
    test_cierre(): GHI = DNI*cos(SZA) + DHI  ->  DNI_calc = (GHI-DHI)/cos(SZA).
    Un patron sostenido del sesgo segun el azimut (en lugar de ruido
    centrado en 0) sugiere una obstruccion del horizonte o un error de
    apuntamiento en una direccion especifica. Cerca de la salida y la
    puesta del sol cos(SZA) es pequeno y el sesgo se vuelve ruidoso aunque
    no exista un problema real; por eso se traza la MEDIANA por intervalo
    de azimut, mas robusta que la dispersion cruda. Solo aplica a fuentes
    con las tres componentes de irradiancia; devuelve None en otro caso.
    """
    req = {"GHI", "DNI", "DHI", "zenith", "azimuth"}
    if not req.issubset(df.columns):
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

    idx = np.random.RandomState(1).choice(len(d), min(len(d), 30000), replace=False)
    muestra = d.iloc[idx]

    # Mediana por intervalo de azimut de 10 grados: revela el patron sin
    # el ruido de la dispersion de fondo.
    d["bin_azimut"] = pd.cut(d["azimuth"], bins=np.arange(0, 361, 10))
    mediana_bin = d.groupby("bin_azimut", observed=True)["sesgo"].median()
    centros = [iv.mid for iv in mediana_bin.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(muestra["azimuth"], muestra["sesgo"], s=4, alpha=0.15,
               color=color, label="Datos (muestra de 30k)")
    ax.plot(centros, mediana_bin.values, color="black", lw=2,
            label="Mediana por bin de 10 grados")
    ax.axhline(0, color=COLOR_FLAG, ls="--", lw=1, label="Sesgo = 0 (cierre perfecto)")
    ax.set_title(titulo)
    ax.set_xlabel("Azimut solar [grados]  (90=Este, 180=Sur, 270=Oeste)")
    ax.set_ylabel("Sesgo DNI = DNI medido - DNI calculado por cierre [W/m2]")
    ax.set_xlim(0, 360)
    ax.legend(fontsize=8, loc="upper right")
    return _guardar(fig, nombre_archivo)


def fig_ktests_envelope(df, titulo_base, nombre_base):
    """
    Visualiza los K-tests (test_ktests, tabla de pruebas del QC experto)
    en dos paneles:
      - K  = DHI/GHI  vs  Kt = GHI/(ETN*cos SZA)
      - Kn = DNI/ETN  vs  Kt
    Colorea cada punto segun el flag YA calculado por test_ktests (columna
    'flag_ktests') para ver donde caen los datos marcados respecto a los
    limites definidos en config.yaml. Los limites de K y Kt son exactos;
    el de Kn se dibuja con el ETN PROMEDIO del periodo, solo como
    referencia visual. Requiere GHI/DNI/DHI y 'flag_ktests' (resultado de
    aplicar_qc_experto con el perfil de la fuente satelital).
    Devuelve [ruta_k_vs_kt, ruta_kn_vs_kt], o None si faltan columnas.
    """
    req = {"GHI", "DNI", "DHI", "zenith", "dni_extra", "flag_ktests"}
    if not req.issubset(df.columns):
        return None

    QE = CONFIG["qc_experto"]
    ALT = CONFIG["sitio"]["altitud_m"]

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

    idx = np.random.RandomState(1).choice(len(d), min(len(d), 30000), replace=False)
    m = d.iloc[idx]
    es_flag = (m["flag_ktests"] == "FLAG").to_numpy()
    colores = np.where(es_flag, COLOR_FLAG, "#9aa0a6")

    rutas = []

    # --- Panel K vs Kt ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(m["Kt"], m["K"], s=5, alpha=0.45, c=colores)
    ax.axvline(QE["kt_max"], color="black", ls="--", lw=1)
    ax.axhline(QE["k_max_sza_baja"], color="#1f5fa3", ls="--", lw=1)
    ax.axhline(QE["k_max_sza_alta"], color="#8a1f1f", ls="--", lw=1)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#9aa0a6", markersize=6, label="OK"),
        Line2D([0], [0], marker="o", linestyle="", color=COLOR_FLAG, markersize=6, label="FLAG"),
        Line2D([0], [0], color="black", ls="--", label=f"Kt max = {QE['kt_max']}"),
        Line2D([0], [0], color="#1f5fa3", ls="--",
               label=f"K max (SZA<{QE['cierre_sza_corte']:.0f}) = {QE['k_max_sza_baja']}"),
        Line2D([0], [0], color="#8a1f1f", ls="--",
               label=f"K max (SZA>={QE['cierre_sza_corte']:.0f}) = {QE['k_max_sza_alta']}"),
    ]
    ax.set_xlabel("Kt = GHI / (ETN * cos SZA)")
    ax.set_ylabel("K = DHI / GHI")
    ax.set_title(f"{titulo_base} - K vs Kt")
    ax.legend(handles=handles, fontsize=8)
    rutas.append(_guardar(fig, f"{nombre_base}_k_vs_kt.png"))

    # --- Panel Kn vs Kt ---
    etn_prom = df["dni_extra"].mean()
    limite_kn_aprox = (QE["kn_factor"] + QE["kn_alt_coef"] * ALT) / etn_prom
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(m["Kt"], m["Kn"], s=5, alpha=0.45, c=colores)
    ax.axvline(QE["kt_max"], color="black", ls="--", lw=1)
    ax.axhline(limite_kn_aprox, color="#1f5fa3", ls="--", lw=1)
    handles2 = [
        Line2D([0], [0], marker="o", linestyle="", color="#9aa0a6", markersize=6, label="OK"),
        Line2D([0], [0], marker="o", linestyle="", color=COLOR_FLAG, markersize=6, label="FLAG"),
        Line2D([0], [0], color="black", ls="--", label=f"Kt max = {QE['kt_max']}"),
        Line2D([0], [0], color="#1f5fa3", ls="--",
               label=f"Kn max aprox. (ETN promedio) = {limite_kn_aprox:.2f}"),
    ]
    ax.set_xlabel("Kt = GHI / (ETN * cos SZA)")
    ax.set_ylabel("Kn = DNI / ETN")
    ax.set_title(f"{titulo_base} - Kn vs Kt")
    ax.legend(handles=handles2, fontsize=8)
    rutas.append(_guardar(fig, f"{nombre_base}_kn_vs_kt.png"))

    return rutas


# ============================================================
#  Figuras de COBERTURA / CONTINUIDAD
#  No requieren QC ni geometria solar, solo la columna 'datetime'.
#  Se usan justo despues de cargar y normalizar los datos, para
#  diagnosticar el estado crudo de cada fuente.
# ============================================================

def fig_continuidad_temporal(df: pd.DataFrame, titulo: str, nombre_archivo: str,
                             cmap: str = "YlGnBu") -> Path:
    """
    Mapa de calor dia del anio (eje Y) vs hora local (eje X): 1 si existe
    AL MENOS un registro en esa hora de ese dia, 0 si no. A diferencia de
    fig_heatmap_diario() (que muestra el VALOR de una variable), esta
    figura muestra solo la EXISTENCIA del dato y revela de un vistazo los
    huecos largos de cobertura (util en estaciones terrestres con huecos).
    """
    d = df[["datetime"]].dropna().copy()
    local = d["datetime"].dt.tz_localize(None) if d["datetime"].dt.tz is not None else d["datetime"]
    anio = int(local.dt.year.mode().iloc[0])
    dias_del_anio = pd.Timestamp(year=anio, month=12, day=31).dayofyear

    d["doy"] = local.dt.dayofyear
    d["hora"] = local.dt.hour
    existe = d.drop_duplicates(["doy", "hora"]).assign(val=1)
    matriz = existe.pivot_table(index="doy", columns="hora", values="val", aggfunc="max", fill_value=0)
    matriz = matriz.reindex(index=range(1, dias_del_anio + 1), columns=range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(matriz.to_numpy(), aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=0, vmax=1, origin="lower",
                   extent=[0, 24, 1, dias_del_anio + 1])
    ax.set_title(titulo)
    ax.set_xlabel("Hora local")
    ax.set_ylabel("Dia del anio")
    ax.set_xticks(range(0, 25, 3))
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1], fraction=0.03, pad=0.02)
    cbar.set_label("Existe registro")
    cbar.ax.set_yticklabels(["No", "Si"])
    return _guardar(fig, nombre_archivo)


def fig_cobertura_mensual(df: pd.DataFrame, titulo: str, nombre_archivo: str,
                          registros_esperados_por_dia: float | None = None,
                          color: str = COLOR_SOLCAST) -> Path:
    """
    Barras de % de cobertura por mes (registros observados / esperados),
    con una linea de referencia al 95%.

    Si no se indica `registros_esperados_por_dia`, se estima a partir del
    INTERVALO de muestreo (1440 min/dia / mediana del intervalo entre
    registros consecutivos), y NO del conteo de registros por dia.
    Los registros esperados se calculan solo sobre los dias observados.
    """
    d = df[["datetime"]].dropna().copy()
    local = d["datetime"].dt.tz_localize(None) if d["datetime"].dt.tz is not None else d["datetime"]
    d["mes"] = local.dt.to_period("M").astype(str)
    d["fecha"] = local.dt.date

    if registros_esperados_por_dia is None:
        intervalo_mediano_min = local.sort_values().diff().dropna().dt.total_seconds().median() / 60
        registros_esperados_por_dia = 1440.0 / intervalo_mediano_min if intervalo_mediano_min else np.nan

    por_mes = d.groupby("mes").agg(
        registros_observados=("fecha", "size"),
        dias_observados=("fecha", "nunique"),
    ).reset_index()
    por_mes["registros_esperados"] = por_mes["dias_observados"] * registros_esperados_por_dia
    por_mes["cobertura_%"] = 100 * por_mes["registros_observados"] / por_mes["registros_esperados"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(por_mes["mes"], por_mes["cobertura_%"], color=color)
    ax.axhline(95, color=COLOR_FLAG, ls="--", lw=1.4, label="Referencia 95%")
    ax.set_title(titulo)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Cobertura [%]")
    ax.set_ylim(0, max(105, por_mes["cobertura_%"].max() * 1.08))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    return _guardar(fig, nombre_archivo)


def fig_histograma_intervalos(df: pd.DataFrame, titulo: str, nombre_archivo: str,
                              color: str = COLOR_ALVERNIA) -> Path:
    """
    Histograma de los intervalos de tiempo entre registros consecutivos
    (en minutos), recortado al percentil 99. Para una fuente perfectamente
    regular (p. ej., una serie satelital cada 5 min) aparece una sola
    barra; para una estacion irregular revela la dispersion real del
    muestreo.
    """
    intervalos = df["datetime"].sort_values().diff().dropna().dt.total_seconds() / 60

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if intervalos.empty:
        ax.text(0.5, 0.5, "Sin intervalos disponibles", ha="center", va="center",
                transform=ax.transAxes)
    else:
        p99 = intervalos.quantile(0.99)
        limite = max(10, p99) if pd.notna(p99) else intervalos.max()
        bins = np.linspace(0, limite, 40)
        ax.hist(intervalos.clip(upper=limite), bins=bins, color=color, edgecolor="white")
    ax.set_title(titulo)
    ax.set_xlabel("Intervalo entre registros [minutos]")
    ax.set_ylabel("Frecuencia")
    return _guardar(fig, nombre_archivo)


# ============================================================
#  Figuras de resultados del site-adaptation
#  Trabajan sobre los CSV ya guardados (predicciones_test,
#  metricas_modelos, importancia_variables), sin necesidad de
#  volver a ejecutar todo el pipeline de QC.
# ============================================================

def fig_dispersion_antes_despues(predicciones_test: pd.DataFrame, target: str,
                                 col_baseline: str, col_mejor_modelo: str,
                                 nombre_mejor_modelo: str, nombre_archivo: str) -> Path:
    """
    Dos paneles lado a lado sobre el conjunto de TEST: fuente satelital
    cruda (`col_baseline`) vs referencia terrestre (`target`) a la
    izquierda, y mejor modelo (`col_mejor_modelo`) vs referencia a la
    derecha. Si el site-adaptation es efectivo, la nube de la derecha se
    concentra mucho mas cerca de la diagonal 1:1 que la de la izquierda.
    """
    d = predicciones_test[[target, col_baseline, col_mejor_modelo]].dropna()
    limite = float(d.max().max()) * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    axes[0].scatter(d[target], d[col_baseline], s=10, alpha=0.45, color=COLOR_SOLCAST)
    axes[0].plot([0, limite], [0, limite], "k--", lw=1.2, label="1:1")
    axes[0].set_title("ANTES: Solcast crudo vs Alvernia")
    axes[0].set_xlabel("GHI Alvernia (referencia) [W/m2]")
    axes[0].set_ylabel("GHI Solcast [W/m2]")
    axes[0].legend(loc="lower right")

    axes[1].scatter(d[target], d[col_mejor_modelo], s=10, alpha=0.45, color="#2a9d8f")
    axes[1].plot([0, limite], [0, limite], "k--", lw=1.2, label="1:1")
    axes[1].set_title(f"DESPUES: {nombre_mejor_modelo} vs Alvernia")
    axes[1].set_xlabel("GHI Alvernia (referencia) [W/m2]")

    for ax in axes:
        ax.set_xlim(0, limite)
        ax.set_ylim(0, limite)

    fig.suptitle("Site-adaptation: antes vs despues (medido SOLO sobre test)", y=1.0)
    return _guardar(fig, nombre_archivo)


def fig_serie_temporal_antes_despues(predicciones_test: pd.DataFrame, target: str,
                                     col_baseline: str, col_mejor_modelo: str,
                                     nombre_mejor_modelo: str, nombre_archivo: str) -> Path:
    """
    Serie temporal con tres curvas superpuestas en el periodo de test:
    la referencia terrestre, la fuente satelital cruda y el mejor modelo
    adaptado. Permite verificar si el modelo sigue mejor la forma real de
    la irradiancia, y no solo su promedio.
    """
    d = predicciones_test[["datetime", target, col_baseline, col_mejor_modelo]].dropna()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(d["datetime"], d[target], color="#1f1f1f", lw=1.3, label="Alvernia (referencia)")
    ax.plot(d["datetime"], d[col_baseline], color=COLOR_SOLCAST, lw=1.0, alpha=0.75, label="Solcast crudo")
    ax.plot(d["datetime"], d[col_mejor_modelo], color="#2a9d8f", lw=1.1, label=f"{nombre_mejor_modelo} (adaptado)")
    ax.set_title("Serie temporal comparativa - periodo de TEST")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("GHI [W/m2]")
    ax.legend()
    return _guardar(fig, nombre_archivo)


def fig_ranking_modelos(tabla_metricas: pd.DataFrame, titulo: str, nombre_archivo: str,
                        color: str = COLOR_SOLCAST, modelo_elegido: str = None) -> Path:
    """
    Barras horizontales de RMSE por modelo, ordenadas de mejor a peor.

    Si se indica `modelo_elegido`, esa barra se resalta con COLOR_ELEGIDO
    (el mismo verde azulado que usan fig_dispersion_antes_despues y
    fig_serie_temporal_antes_despues para el modelo adaptado). Esto es
    util cuando el modelo seleccionado NO es el de menor RMSE porque otros
    se descartaron por el filtro de consistencia fisica/estructural: la
    figura deja claro a simple vista cual se utiliza.
    """
    d = tabla_metricas.sort_values("RMSE", ascending=True)
    colores = [COLOR_ELEGIDO if m == modelo_elegido else color for m in d["modelo"]]

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(d) + 1.5))
    ax.barh(d["modelo"], d["RMSE"], color=colores)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE [W/m2] (sobre test)")
    ax.set_title(titulo)
    for i, valor in enumerate(d["RMSE"]):
        ax.text(valor, i, f" {valor:.1f}", va="center")

    if modelo_elegido is not None and modelo_elegido in d["modelo"].values:
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=COLOR_ELEGIDO, label=f"Modelo elegido: {modelo_elegido}"),
            Patch(color=color, label="Otros modelos evaluados"),
        ], loc="upper right", fontsize=9)

    return _guardar(fig, nombre_archivo)

def fig_importancia_variables(tabla_importancia: pd.DataFrame, titulo: str, nombre_archivo: str,
                              color: str = COLOR_ALVERNIA) -> Path:
    """Barras horizontales de importancia relativa de variables (modelos
    basados en arboles), ordenadas de menor a mayor."""
    d = tabla_importancia.sort_values("importancia", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(d) + 1.5))
    ax.barh(d["variable"], d["importancia"], color=color)
    ax.set_xlabel("Importancia relativa")
    ax.set_title(titulo)
    return _guardar(fig, nombre_archivo)
