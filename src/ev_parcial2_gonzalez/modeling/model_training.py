"""
Módulo de entrenamiento de modelos supervisados.

Define los 10 clasificadores obligatorios, el clasificador opcional XGBoost,
los 2 regresores recomendados, y las funciones de construcción de pipelines
con SMOTE para balanceo de clases.
"""

from typing import Dict

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ev_parcial2_gonzalez.modeling.config import (
    N_JOBS,
    RANDOM_STATE,
    SMOTE_SAMPLING_STRATEGY,
)
from ev_parcial2_gonzalez.modeling.utils import log_execution, setup_logger

logger = setup_logger(__name__)

# Attempt to import XGBClassifier; if unavailable, define a placeholder.
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None  # type: ignore[assignment, misc]
    logger.warning("XGBoost is not installed; XGBClassifier will be unavailable.")


# ---------------------------------------------------------------------------
# Classifier catalog
# ---------------------------------------------------------------------------

def get_supervised_classifiers() -> Dict[str, ClassifierMixin]:
    """
    Retorna un diccionario con los **10 clasificadores obligatorios** y
    el clasificador extra ``XGBClassifier`` (si está instalado).

    Todos los modelos se instancian con ``random_state=42`` donde aplique
    para garantizar reproducibilidad absoluta.

    Returns:
        Dict[str, ClassifierMixin]: Clasificadores listos para entrenar.
    """
    classifiers: Dict[str, ClassifierMixin] = {
        "logistic_regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=50, random_state=RANDOM_STATE
        ),
        "svc": SVC(probability=True, cache_size=1000, max_iter=200, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(),
        "gaussian_nb": GaussianNB(),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        "adaboost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=200,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }

    # Conditionally include XGBoost
    if XGBClassifier is not None:
        classifiers["xgboost"] = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )

    return classifiers


# ---------------------------------------------------------------------------
# Regressor catalog
# ---------------------------------------------------------------------------

def get_regressors() -> Dict[str, RegressorMixin]:
    """
    Retorna un diccionario con los regresores recomendados para alcanzar
    el 100 % de la rúbrica.

    Returns:
        Dict[str, RegressorMixin]: Regresores listos para entrenar.
    """
    return {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_classification_pipeline(
    preprocessor: ColumnTransformer,
    classifier: ClassifierMixin,
) -> ImbPipeline:
    """
    Construye un ``ImbPipeline`` con preprocesamiento, SMOTE y clasificador.

    El pipeline sigue el orden canónico:
    ``preprocessor → SMOTE → classifier``.

    Args:
        preprocessor: ``ColumnTransformer`` de preprocesamiento.
        classifier: Clasificador de scikit-learn.

    Returns:
        ImbPipeline: Pipeline listo para ``.fit()`` / ``.predict()``.
    """
    return ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                    sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def build_regression_pipeline(
    preprocessor: ColumnTransformer,
    regressor: RegressorMixin,
) -> ImbPipeline:
    """
    Construye un pipeline de regresión (sin SMOTE).

    Args:
        preprocessor: ``ColumnTransformer`` de preprocesamiento.
        regressor: Regresor de scikit-learn.

    Returns:
        ImbPipeline: Pipeline de regresión.
    """
    return ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


# ---------------------------------------------------------------------------
# Batch training
# ---------------------------------------------------------------------------

@log_execution
def train_all_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Dict[str, ImbPipeline]:
    """
    Entrena **todos** los clasificadores supervisados usando SMOTE.

    Args:
        X_train: Matriz de características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        preprocessor: ``ColumnTransformer`` preconfigurado.

    Returns:
        Dict[str, ImbPipeline]: Pipelines entrenados indexados por nombre.
    """
    trained: Dict[str, ImbPipeline] = {}
    classifiers = get_supervised_classifiers()

    for name, clf in classifiers.items():
        try:
            pipe = build_classification_pipeline(preprocessor, clf)
            pipe.fit(X_train, y_train)
            trained[name] = pipe
            logger.info(f"Clasificador entrenado: {name}")
        except Exception as e:
            logger.error(f"Error entrenando {name}: {e}", exc_info=True)

    return trained


@log_execution
def train_all_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Dict[str, ImbPipeline]:
    """
    Entrena todos los regresores definidos (sin SMOTE).

    Args:
        X_train: Matriz de características de entrenamiento.
        y_train: Variable objetivo continua.
        preprocessor: ``ColumnTransformer`` preconfigurado.

    Returns:
        Dict[str, ImbPipeline]: Pipelines de regresión entrenados.
    """
    trained: Dict[str, ImbPipeline] = {}
    regressors = get_regressors()

    for name, reg in regressors.items():
        try:
            pipe = build_regression_pipeline(preprocessor, reg)
            pipe.fit(X_train, y_train)
            trained[name] = pipe
            logger.info(f"Regresor entrenado: {name}")
        except Exception as e:
            logger.error(f"Error entrenando regresor {name}: {e}", exc_info=True)

    return trained
