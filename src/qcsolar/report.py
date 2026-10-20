"""
report.py
---------
Gestion de salidas del paquete. Centraliza tres tareas:

  1. Guardar datasets intermedios (datos transformados, en CSV) en
     data/interim/ o data/processed/.
  2. Guardar tablas de documentacion (transformaciones, estadisticas,
     conteos) en outputs/tablas/.
  3. Proveer funciones para construir esas tablas.

Cada etapa del pipeline LEE de la carpeta de la etapa anterior y ESCRIBE
un dataset nuevo junto con sus tablas, lo que deja trazabilidad completa
de como se transformo la informacion original.
"""

from pathlib import Path
import pandas as pd

from .config import resolve_path, CONFIG


# --- Carpetas de salida (se crean si no existen) ---
DIR_INTERIM = resolve_path(CONFIG["salidas"]["interim"])
DIR_PROCESSED = resolve_path(CONFIG["salidas"]["processed"])
DIR_OUTPUTS = resolve_path(CONFIG["salidas"]["outputs"])
DIR_TABLAS = DIR_OUTPUTS / "tablas"
DIR_FIGURAS = DIR_OUTPUTS / "figuras"

for carpeta in [DIR_INTERIM, DIR_PROCESSED, DIR_OUTPUTS, DIR_TABLAS, DIR_FIGURAS]:
    carpeta.mkdir(parents=True, exist_ok=True)


# ============================================================
#  Guardar datasets
# ============================================================

def guardar_dataset(df: pd.DataFrame, nombre: str, etapa: str = "interim") -> Path:
    """
    Guarda un DataFrame como CSV (UTF-8 con BOM) en la carpeta de la etapa
    indicada y devuelve la ruta del archivo.

    etapa: "interim"   -> data/interim/   (datos normalizados)
           "processed" -> data/processed/ (datos con geometria, flags, etc.)
    """
    carpeta = DIR_INTERIM if etapa == "interim" else DIR_PROCESSED
    ruta = carpeta / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"   [dataset guardado] {ruta.relative_to(resolve_path('.'))}  ({len(df):,} filas)")
    return ruta


# ============================================================
#  Guardar tablas de documentacion
# ============================================================

def guardar_tabla(df: pd.DataFrame, nombre: str) -> Path:
    """Guarda una tabla de documentacion como CSV en outputs/tablas/ y
    devuelve la ruta del archivo."""
    ruta = DIR_TABLAS / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"   [tabla guardada]   {ruta.relative_to(resolve_path('.'))}")
    return ruta


# ============================================================
#  Constructores de tablas utiles
# ============================================================

def tabla_estadisticas(df: pd.DataFrame, columnas: list[str], etiqueta: str = "") -> pd.DataFrame:
    """
    Resumen estadistico de columnas numericas: n, faltantes (absolutos y
    %), media, mediana, min, max y desviacion estandar. Documenta la
    distribucion de los datos en un punto del pipeline; `etiqueta`
    identifica la etapa y las columnas inexistentes se omiten.
    """
    filas = []
    for col in columnas:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        filas.append({
            "etapa": etiqueta,
            "variable": col,
            "n": int(x.notna().sum()),
            "faltantes": int(x.isna().sum()),
            "faltantes_%": round(100 * x.isna().mean(), 3),
            "media": round(float(x.mean()), 3),
            "mediana": round(float(x.median()), 3),
            "min": round(float(x.min()), 3),
            "max": round(float(x.max()), 3),
            "std": round(float(x.std()), 3),
        })
    return pd.DataFrame(filas)


def tabla_transformacion(filas: list[dict]) -> pd.DataFrame:
    """
    Construye el registro de transformaciones a partir de una lista de
    diccionarios con las claves: elemento, antes, despues, accion.
    """
    return pd.DataFrame(filas, columns=["elemento", "antes", "despues", "accion"])


def tabla_conteo_registros(pasos: list[dict]) -> pd.DataFrame:
    """
    'Balance de materia' del dato: cuantos registros quedan en cada paso.
    Cada paso es un dict con: paso, registros, descripcion. Agrega la
    columna 'cambio' con la diferencia respecto al paso anterior.
    """
    df = pd.DataFrame(pasos, columns=["paso", "registros", "descripcion"])
    # Variacion de registros respecto al paso anterior (0 en el primero)
    df["cambio"] = df["registros"].diff().fillna(0).astype(int)
    return df
