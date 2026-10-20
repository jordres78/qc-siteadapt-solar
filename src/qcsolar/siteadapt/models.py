"""
siteadapt/models.py
--------------------
Modelos candidatos de site-adaptation: desde el baseline trivial (fuente
satelital sin corregir) hasta modelos de aprendizaje automático. Todos
siguen la MISMA interfaz (fit(X, y) / predict(X)) para entrenarlos y
evaluarlos en un solo bucle, sin código especial por modelo.

Los nueve candidatos (de simple a complejo) se comparan en igualdad de
condiciones; ninguno se excluye a priori por su familia:
  1. Baseline (satelital sin corregir) -- la referencia a superar.
  2. Regresión lineal simple (GHI satelital -> GHI terrestre) -- el
     site-adaptation clásico más simple e interpretable.
  3. Regresión lineal multivariable (todas las predictoras) -- referencia
     intermedia entre el modelo simple y los modelos no lineales.
  4. EQM (Empirical Quantile Mapping) -- método clásico de la literatura
     de site-adaptation. Corrige el sesgo Y la forma de la distribución,
     no solo la media (a diferencia de la regresión lineal).
  5-7. Random Forest, Extra Trees, Gradient Boosting -- ensambles de
     árboles de scikit-learn.
  8. XGBoost -- gradient boosting regularizado; en la literatura de
     site-adaptation con insumos satelitales + terrestres ha reducido el
     MAE frente a los métodos tradicionales.
  9. Red neuronal (MLP, scikit-learn) -- perceptrón multicapa con
     estandarización previa de variables (a diferencia de los árboles,
     las redes SÍ son sensibles a la escala de cada predictora). Se usa
     MLPRegressor con pérdida MSE estándar para no añadir una dependencia
     de deep learning (TensorFlow/PyTorch) solo por este modelo.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class ModeloBaseline:
    """Modelo nulo: devuelve la GHI satelital sin corregir (el estado "antes")."""
    nombre = "Baseline (Solcast sin corregir)"

    def fit(self, X, y):
        """No aprende nada; existe solo para respetar la interfaz común."""
        return self

    def predict(self, X):
        """Devuelve la columna de GHI satelital de `X` como predicción."""
        return X["GHI_solcast"].to_numpy()


class ModeloLinealSimple:
    """Regresión lineal de una sola variable: GHI terrestre ~ GHI satelital."""
    nombre = "Regresion lineal simple"

    def __init__(self):
        """Crea el regresor lineal subyacente."""
        self.modelo = LinearRegression()

    def fit(self, X, y):
        """Ajusta la recta usando solo la GHI satelital como predictora."""
        self.modelo.fit(X[["GHI_solcast"]], y)
        return self

    def predict(self, X):
        """Predice la GHI terrestre a partir de la GHI satelital de `X`."""
        return self.modelo.predict(X[["GHI_solcast"]])


class ModeloLinealMultivariable:
    """Regresión lineal con todas las predictoras disponibles (satelitales + temporales)."""
    nombre = "Regresion lineal multivariable"

    def __init__(self, columnas_predictoras):
        """`columnas_predictoras`: columnas de `X` que usa el modelo."""
        self.columnas = columnas_predictoras
        self.modelo = LinearRegression()

    def fit(self, X, y):
        """Ajusta la regresión sobre las columnas predictoras de `X`."""
        self.modelo.fit(X[self.columnas], y)
        return self

    def predict(self, X):
        """Predice la GHI terrestre con las columnas predictoras de `X`."""
        return self.modelo.predict(X[self.columnas])


class ModeloEQM:
    """
    Empirical Quantile Mapping: mapea cada valor de GHI satelital al valor
    de GHI terrestre que tiene el MISMO percentil dentro de sus respectivas
    distribuciones de ENTRENAMIENTO. Un dato nuevo fuera del rango visto en
    entrenamiento se extrapola de forma constante (límite superior/inferior
    de la tabla de mapeo).
    """
    nombre = "EQM (Quantile Mapping)"

    def __init__(self, n_quantiles=100):
        """`n_quantiles`: número de intervalos de la tabla de mapeo (n_quantiles + 1 percentiles)."""
        self.n_quantiles = n_quantiles
        self.q_solcast_ = None
        self.q_alvernia_ = None

    def fit(self, X, y):
        """Calcula los percentiles de la GHI satelital y de la terrestre en entrenamiento."""
        percentiles = np.linspace(0, 100, self.n_quantiles + 1)
        x = X["GHI_solcast"].dropna()
        y_ = pd.Series(y).dropna()
        self.q_solcast_ = np.percentile(x, percentiles)
        self.q_alvernia_ = np.percentile(y_, percentiles)
        return self

    def predict(self, X):
        """Interpola cada GHI satelital de `X` en la tabla de percentiles."""
        x = X["GHI_solcast"].to_numpy()
        return np.interp(x, self.q_solcast_, self.q_alvernia_)


class _EnvolturaSklearn:
    """Envoltura genérica para que un regresor de scikit-learn (o compatible) use la interfaz común fit/predict."""

    def __init__(self, modelo_sklearn, columnas, nombre):
        """Guarda el regresor, las columnas predictoras que usa y su nombre."""
        self.modelo = modelo_sklearn
        self.columnas = columnas
        self.nombre = nombre

    def fit(self, X, y):
        """Entrena el regresor envuelto sobre las columnas predictoras de `X`."""
        self.modelo.fit(X[self.columnas], y)
        return self

    def predict(self, X):
        """Predice con el regresor envuelto sobre las columnas predictoras de `X`."""
        return self.modelo.predict(X[self.columnas])

    @property
    def feature_importances_(self):
        """Importancia de variables del regresor envuelto, o None si no la expone."""
        return getattr(self.modelo, "feature_importances_", None)


def construir_modelos(columnas_predictoras: list[str], random_state: int = 42) -> dict:
    """
    Devuelve un diccionario {nombre: modelo} con TODOS los modelos a
    comparar, listos para .fit(X_train, y_train) / .predict(X_nuevo).

    `columnas_predictoras` es la lista completa de variables disponibles
    (la usan el modelo multivariable y los de aprendizaje automático; el
    baseline, la regresión simple y el EQM solo usan la GHI satelital).
    `random_state` fija la semilla para que los resultados sean reproducibles.
    """
    modelos = {
        "Baseline (Solcast sin corregir)": ModeloBaseline(),
        "Regresion lineal simple": ModeloLinealSimple(),
        "Regresion lineal multivariable": ModeloLinealMultivariable(columnas_predictoras),
        "EQM (Quantile Mapping)": ModeloEQM(),
        "Random Forest": _EnvolturaSklearn(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=random_state, n_jobs=-1),
            columnas_predictoras, "Random Forest"),
        "Extra Trees": _EnvolturaSklearn(
            ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                 random_state=random_state, n_jobs=-1),
            columnas_predictoras, "Extra Trees"),
        "Gradient Boosting": _EnvolturaSklearn(
            GradientBoostingRegressor(random_state=random_state),
            columnas_predictoras, "Gradient Boosting"),
        "XGBoost": _EnvolturaSklearn(
            XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=random_state, n_jobs=-1),
            columnas_predictoras, "XGBoost"),
        "Red neuronal (MLP)": _EnvolturaSklearn(
            Pipeline([
                ("escalado", StandardScaler()),
                ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                      solver="adam", alpha=1e-4, max_iter=1000,
                                      early_stopping=True, n_iter_no_change=15,
                                      random_state=random_state)),
            ]),
            columnas_predictoras, "Red neuronal (MLP)"),
    }
    return modelos
