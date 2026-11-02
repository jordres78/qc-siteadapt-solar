"""
mosaico.py
----------
Mosaico de diagnóstico visual del control de calidad, al estilo de la
Fig. 1 de Forstinger et al. (2021). Seis paneles con la GHI de una serie:

  1. GHI frente a la elevación solar con los límites BSRN.
  2. Patrón día x hora de la GHI media.
  3. Envolvente de la GHI máxima en el espacio elevación-azimut.
  4. Índice de cielo despejado (CSI) a lo largo del año.
  5. Dispersión de la serie frente a la fuente satelital.
  6. Índice de claridad Kt frente a la elevación.

El mismo mosaico se dibuja dos veces, con el dato crudo y con la serie
reconstruida y verificada, para comparar el antes y el después de la
reconstrucción por site-adaptation y del nuevo control de calidad.

Ambas versiones se comparan a 5 min, la resolución nativa de la fuente
satelital. El dato crudo de 5 min es la columna GHI_original de
*_ghi_corregido.csv, que es el promedio de 5 min de todos los registros
medidos, sin filtrar (para Udenar 2025, el promedio de los registros de
1 min). La serie final es la variante elegida (total o selectiva) en los
instantes que superan el nuevo control de calidad.

El módulo solo depende de numpy, pandas y matplotlib, de modo que la
aplicación web, la presentación y el artículo dibujan exactamente la
misma figura sin importar el resto del paquete.
"""
from pathlib import Path
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Archivo satelital que acompaña a cada conjunto de medición en tierra
SATELITAL_POR_CLAVE = {
    "udenar": "solcast_qc_completo.csv",            # Udenar 2023 con Solcast 2023
    "udenar_est": "solcast_2025_qc_completo.csv",   # Udenar 2025 con Solcast 2025
}

CMAP_SOLAR = LinearSegmentedColormap.from_list("solar", [
    (0, "#f7fbff"), (.12, "#c6dbef"), (.28, "#6baed6"), (.44, "#3288bd"),
    (.58, "#66c2a5"), (.70, "#ffffbf"), (.82, "#fdae61"), (.92, "#f46d43"),
    (1, "#d73027")])
CMAP_DENSIDAD = LinearSegmentedColormap.from_list("dens", ["#c6dbef", "#3288bd", "#1a237e"])
VERDE_NUMERO = "#00923f"
E0_MEDIA = 1361.0
_COMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


# ------------------------------------------------------------------
#  Lectura y armado de las dos series
# ------------------------------------------------------------------
def leer_procesado(ruta: Path, columnas=None) -> pd.DataFrame:
    """Lee un CSV de data/processed y deja la fecha en hora local sin zona."""
    d = pd.read_csv(ruta, usecols=columnas)
    fecha = pd.to_datetime(d["datetime"], utc=True)
    d["datetime"] = fecha.dt.tz_convert("America/Bogota").dt.tz_localize(None)
    return d.sort_values("datetime").reset_index(drop=True)


def series_antes_despues(dir_processed: Path, clave: str, variante: str = "selectiva"):
    """Devuelve (antes, despues), dos DataFrames a 5 min con las columnas
    datetime, GHI, SAT, elevation, azimuth, dni_extra y ghi_clear.

    antes   -- dato crudo promediado a 5 min (GHI_original), sin filtrar.
    despues -- variante `variante` de la serie reconstruida, solo en los
               instantes con usabilidad OK en el nuevo control de calidad.
    La geometría solar y la GHI satelital (SAT) vienen de la fuente
    satelital del mismo año, con la misma marca de tiempo."""
    dir_processed = Path(dir_processed)
    # En los CSV las columnas usan los sufijos "total" y "selectivo"
    sufijo = {"selectiva": "selectivo", "selectivo": "selectivo", "total": "total"}[variante]
    geo = leer_procesado(dir_processed / SATELITAL_POR_CLAVE[clave],
                         ["datetime", "GHI", "elevation", "azimuth", "dni_extra", "ghi_clear"])
    geo = geo.rename(columns={"GHI": "SAT"})
    rec = leer_procesado(dir_processed / f"{clave}_ghi_corregido.csv",
                         ["datetime", "GHI_original", f"GHI_corregido_{sufijo}",
                          f"usabilidad_{sufijo}"])
    rec = rec.merge(geo, on="datetime", how="left")
    base = ["datetime", "SAT", "elevation", "azimuth", "dni_extra", "ghi_clear"]
    antes = rec[base].assign(GHI=rec["GHI_original"])
    despues = rec[base].assign(
        GHI=rec[f"GHI_corregido_{sufijo}"].where(rec[f"usabilidad_{sufijo}"].eq("OK")))
    return antes, despues


# ------------------------------------------------------------------
#  Conteos de los problemas que muestra el mosaico
# ------------------------------------------------------------------
def conteos_mosaico(df: pd.DataFrame) -> dict:
    """Registros con dato, nocturnos con GHI > 10, sobre el límite físico
    BSRN y con CSI > 1,35 (cuando GHI y GHI de cielo despejado > 50)."""
    d = df.dropna(subset=["GHI", "elevation"])
    mu = np.cos(np.radians(90 - d["elevation"].clip(lower=0)))
    limite = 1.5 * d["dni_extra"] * mu ** 1.2 + 100
    c = d[(d["ghi_clear"] > 50) & (d["GHI"] > 50)]
    return {
        "registros": int(len(d)),
        "noche_mayor_10": int(((d["elevation"] <= 0) & (d["GHI"] > 10)).sum()),
        "sobre_limite_fisico": int((d["GHI"] > limite).sum()),
        "csi_mayor_135": int((c["GHI"] / c["ghi_clear"] > 1.35).sum()),
    }


# ------------------------------------------------------------------
#  Figura
# ------------------------------------------------------------------
def fig_mosaico_qc(df: pd.DataFrame, titulo: str | None = None, figsize=(7.2, 4.3),
                   tam_letra: float = 8.0, vmax_hora: float = 900, vmax_env: float = 1200):
    """Dibuja el mosaico de seis paneles de una serie y devuelve la figura.

    `df` debe traer datetime, GHI, SAT, elevation, azimuth, dni_extra y
    ghi_clear. Las escalas de color son fijas para que el mosaico crudo y
    el final se puedan comparar panel a panel."""
    estilo = {
        "font.family": "DejaVu Sans", "font.size": tam_letra,
        "axes.labelsize": tam_letra + .5, "axes.titlesize": tam_letra + 1,
        "xtick.labelsize": tam_letra - .5, "ytick.labelsize": tam_letra - .5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
        "axes.grid": False,
    }
    d = df.dropna(subset=["GHI", "elevation"]).copy()
    with plt.rc_context(estilo):
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = fig.add_gridspec(2, 3)
        axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

        # 1. GHI frente a la elevación con los límites BSRN
        ax = axs[0]
        ax.hexbin(d["elevation"], d["GHI"], gridsize=60, extent=(-30, 90, -50, 1600),
                  cmap=CMAP_DENSIDAD, bins="log", mincnt=1, linewidths=0)
        el = np.linspace(0, 90, 200)
        m = np.cos(np.radians(90 - el))
        ax.plot(el, 1.5 * E0_MEDIA * m ** 1.2 + 100, c="#c62828", lw=1, label="Límite físico")
        ax.plot(el, 1.2 * E0_MEDIA * m ** 1.2 + 50, c="#ef6c00", lw=1, ls="--",
                label="Valor poco frecuente")
        ax.axvline(0, c=".3", lw=.6, ls=":")
        ax.set(xlim=(-30, 90), ylim=(-50, 1600), xticks=[-25, 0, 25, 50, 75],
               xlabel="Elevación solar [°]", ylabel="GHI [W/m²]")
        ax.legend(loc="upper left", fontsize=tam_letra - 1.5, frameon=False)

        # 2. Patrón día x hora
        ax = axs[1]
        a = d.assign(h=d["datetime"].dt.hour + d["datetime"].dt.minute / 60,
                     dia=d["datetime"].dt.dayofyear)
        a["hb"] = (a["h"] * 2).round() / 2
        p = a.groupby(["hb", "dia"])["GHI"].mean().unstack().reindex(columns=range(1, 367))
        ax.pcolormesh(p.columns, p.index, p.to_numpy(), cmap=CMAP_SOLAR, vmin=0, vmax=vmax_hora,
                      shading="nearest", rasterized=True)
        ax.set(xlim=(1, 366), ylim=(0, 24), yticks=[0, 6, 12, 18, 24], xlabel="Día del año",
               ylabel="Hora local")

        # 3. Envolvente elevación-azimut
        ax = axs[2]
        b = d[d["elevation"] > -5].copy()
        b["el"] = b["elevation"] // 1
        b["az"] = (b["azimuth"] // 2) * 2
        q = b.groupby(["el", "az"])["GHI"].max().unstack()
        im = ax.pcolormesh(q.columns, q.index, q.to_numpy(), cmap=CMAP_SOLAR, vmin=0, vmax=vmax_env,
                           shading="nearest", rasterized=True)
        ax.set(xlim=(0, 360), ylim=(-5, 90), xticks=[0, 90, 180, 270, 360], yticks=[0, 30, 60, 90],
               xlabel="Azimut [°]", ylabel="Elevación [°]")
        cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.02)
        cb.set_label("GHI máx. [W/m²]", fontsize=tam_letra - 1)
        cb.ax.tick_params(labelsize=tam_letra - 1.5)

        # 4. CSI a lo largo del año
        ax = axs[3]
        c = d[(d["ghi_clear"] > 50) & (d["GHI"] > 50)]
        csi = c["GHI"] / c["ghi_clear"]
        ax.scatter(c["datetime"].dt.dayofyear, csi, s=.4, c="#3288bd", alpha=.2, lw=0,
                   rasterized=True)
        ax.axhline(1.35, c="#c62828", lw=1)
        ax.axhline(1, c=".4", lw=.6, ls="--")
        ax.text(5, 1.4, "Umbral 1,35", color="#c62828", fontsize=tam_letra - 1.5)
        ax.set(xlim=(1, 366), ylim=(0, 2), xlabel="Día del año", ylabel="CSI = GHI / GHI$_{cs}$")
        ax.yaxis.set_major_formatter(_COMA)

        # 5. Dispersión frente a la fuente satelital
        ax = axs[4]
        pr = d.dropna(subset=["SAT"])
        ax.hexbin(pr["SAT"], pr["GHI"], gridsize=55, extent=(0, 1300, 0, 1300), cmap=CMAP_DENSIDAD,
                  bins="log", mincnt=1, linewidths=0)
        ax.plot([0, 1300], [0, 1300], c=".3", lw=.8, ls="--")
        ax.set(xlim=(0, 1300), ylim=(0, 1300), xticks=[0, 400, 800, 1200],
               yticks=[0, 400, 800, 1200], xlabel="GHI satelital [W/m²]", ylabel="GHI serie [W/m²]")
        ax.set_aspect("equal", adjustable="box")

        # 6. Índice de claridad frente a la elevación
        ax = axs[5]
        k = d[d["elevation"] > 2]
        kt = k["GHI"] / (k["dni_extra"] * np.cos(np.radians(90 - k["elevation"])))
        ax.hexbin(k["elevation"], kt, gridsize=55, extent=(0, 90, 0, 2), cmap=CMAP_DENSIDAD,
                  bins="log", mincnt=1, linewidths=0)
        ax.axhline(1.35, c="#c62828", lw=1)
        ax.set(xlim=(0, 90), ylim=(0, 2), xticks=[0, 30, 60, 90], xlabel="Elevación solar [°]",
               ylabel="Kt [–]")
        ax.yaxis.set_major_formatter(_COMA)

        # Número de cada panel en un círculo verde, encima del área de la gráfica
        for i, ax in enumerate(axs, 1):
            ax.text(1.0, 1.03, str(i), transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=tam_letra + .5, fontweight="bold", color="white", clip_on=False,
                    bbox=dict(boxstyle="circle,pad=0.22", fc=VERDE_NUMERO, ec="none"))
        if titulo:
            fig.suptitle(titulo, fontsize=tam_letra + 2, fontweight="bold")
    return fig


def png_mosaico(df: pd.DataFrame, titulo: str | None = None, dpi: int = 200, **kw) -> bytes:
    """Dibuja el mosaico y lo devuelve como PNG en memoria (para la app)."""
    fig = fig_mosaico_qc(df, titulo, **kw)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return buf.getvalue()
