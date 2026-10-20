"""
config.py
---------
Carga config.yaml y expone la configuracion del paquete.
Resuelve todas las rutas de forma RELATIVA a la raiz del proyecto, para
que el codigo funcione en cualquier equipo sin rutas absolutas fijas.
Expone ademas atajos a los parametros del sitio (latitud, longitud,
altitud y zona horaria).
"""

from pathlib import Path
import yaml


# La raiz del proyecto es la carpeta que contiene config.yaml.
# __file__ es este archivo (src/qcsolar/config.py); se suben 3 niveles.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_name: str = "config.yaml") -> dict:
    """Carga el archivo de configuracion (por defecto config.yaml, en la
    raiz del proyecto) como diccionario; FileNotFoundError si no existe."""
    config_path = PROJECT_ROOT / config_name
    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontro {config_path}. "
            f"Asegurate de que config.yaml este en la raiz del proyecto."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(relative_path: str) -> Path:
    """Convierte una ruta relativa de la configuracion en una ruta absoluta
    respecto a la raiz del proyecto."""
    return PROJECT_ROOT / relative_path


# Carga la configuracion una sola vez al importar el modulo.
CONFIG = load_config()

# Atajos a los valores del sitio mas utilizados.
SITIO = CONFIG["sitio"]
LATITUD = SITIO["latitud"]
LONGITUD = SITIO["longitud"]
ALTITUD_M = SITIO["altitud_m"]
TIMEZONE = SITIO["timezone"]
