"""
Módulo de optimización de hiperparámetros.

Implementa búsqueda sistemática mediante ``GridSearchCV`` y
``RandomizedSearchCV`` para los modelos campeones detectados
en la etapa de evaluación.  Incluye comparación Before vs After.
"""

import os
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from ev_parcial2_gonzalez.modeling.config import (
    CV_SPLITS,
    N_JOBS,
    RANDOM_STATE,
    SCORING_METRIC,
)
from ev_parcial2_gonzalez.modeling.utils import log_execution, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "classifier__n_estimators": [100, 300, 500],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    },
    "extra_trees": {
        "classifier__n_estimators": [100, 300, 500],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__subsample": [0.8, 1.0],
    },
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "saga"],
    },
    "svc": {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale", "auto"],
    },
    "xgboost": {
        "classifier__n_estimators": [100, 300, 500],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__subsample": [0.8, 1.0],
    },
    "gaussian_nb": {
        "classifier__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    },
    "knn": {
        "classifier__n_neighbors": [3, 5, 7, 11, 15],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    },
}


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

@log_execution
def run_grid_search(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, Any],
    scoring: str = SCORING_METRIC,
    cv_splits: int = CV_SPLITS,
) -> GridSearchCV:
    """
    Ejecuta ``GridSearchCV`` sobre un pipeline dado.

    Args:
        pipeline: Pipeline de ``imblearn`` o ``sklearn``.
        X_train: Datos de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        param_grid: Grilla de hiperparámetros.
        scoring: Métrica de optimización (default: ``f1_macro``).
        cv_splits: Número de folds de CV.

    Returns:
        GridSearchCV: Objeto ajustado con los mejores parámetros.
    """
    skf = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE
    )
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=skf,
        n_jobs=N_JOBS,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    logger.info(f"Mejor score ({scoring}): {gs.best_score_:.4f}")
    logger.info(f"Mejores parámetros: {gs.best_params_}")
    return gs


# ---------------------------------------------------------------------------
# Randomized search
# ---------------------------------------------------------------------------

@log_execution
def run_randomized_search(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_distributions: Dict[str, Any],
    n_iter: int = 50,
    scoring: str = SCORING_METRIC,
    cv_splits: int = CV_SPLITS,
) -> RandomizedSearchCV:
    """
    Ejecuta ``RandomizedSearchCV`` sobre un pipeline dado.

    Args:
        pipeline: Pipeline de ``imblearn`` o ``sklearn``.
        X_train: Datos de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        param_distributions: Distribuciones de hiperparámetros.
        n_iter: Número de combinaciones aleatorias a probar.
        scoring: Métrica de optimización (default: ``f1_macro``).
        cv_splits: Número de folds de CV.

    Returns:
        RandomizedSearchCV: Objeto ajustado con los mejores parámetros.
    """
    skf = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE
    )
    rs = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=skf,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    rs.fit(X_train, y_train)
    logger.info(f"Mejor score ({scoring}): {rs.best_score_:.4f}")
    logger.info(f"Mejores parámetros: {rs.best_params_}")
    return rs


# ---------------------------------------------------------------------------
# Before vs After comparison
# ---------------------------------------------------------------------------

@log_execution
def compare_before_after(
    base_results: pd.DataFrame,
    tuned_model_name: str,
    tuned_score: float,
    metric_col: str = "F1-Score_mean",
) -> pd.DataFrame:
    """
    Genera la tabla comparativa **Before vs After** de tuning.

    Args:
        base_results: DataFrame con resultados base de CV (raw).
        tuned_model_name: Nombre del modelo optimizado.
        tuned_score: F1-Score del modelo post-tuning.
        metric_col: Columna de métrica a comparar.

    Returns:
        pd.DataFrame: Tabla con columnas ``Model``, ``F1 Base``,
            ``F1 Tuned``, ``Improvement``.
    """
    rows = []
    for _, row in base_results.iterrows():
        name = row["Model"]
        base_f1 = row.get(metric_col, 0.0)
        if name == tuned_model_name:
            rows.append({
                "Model": name,
                "F1 Base": f"{base_f1:.4f}",
                "F1 Tuned": f"{tuned_score:.4f}",
                "Improvement": f"{tuned_score - base_f1:+.4f}",
            })
        else:
            rows.append({
                "Model": name,
                "F1 Base": f"{base_f1:.4f}",
                "F1 Tuned": "—",
                "Improvement": "—",
            })
    return pd.DataFrame(rows)


def get_param_grid(model_name: str) -> Dict[str, Any]:
    """
    Retorna la grilla de hiperparámetros predefinida para un modelo.

    Args:
        model_name: Nombre del modelo (ej. ``'random_forest'``).

    Returns:
        Dict[str, Any]: Grilla de parámetros.

    Raises:
        KeyError: Si el modelo no tiene grilla predefinida.
    """
    if model_name not in PARAM_GRIDS:
        raise KeyError(
            f"No hay grilla predefinida para '{model_name}'. "
            f"Opciones: {list(PARAM_GRIDS.keys())}"
        )
    return PARAM_GRIDS[model_name]
