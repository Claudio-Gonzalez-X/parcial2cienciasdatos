"""
Módulo de evaluación de modelos supervisados.

Implementa validación cruzada estratificada con ``StratifiedKFold``,
calcula las 5 métricas obligatorias (Accuracy, Precision, Recall,
F1-Score, ROC-AUC) con sus estadísticas de estabilidad (mean ± std),
y exporta los resultados a CSV.
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from ev_parcial2_gonzalez.modeling.config import (
    CV_SPLITS,
    RANDOM_STATE,
    REPORTING_DIR,
    SCORING_METRIC,
)
from ev_parcial2_gonzalez.modeling.utils import log_execution, setup_logger

logger = setup_logger(__name__)


def _evaluate_fold(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    pipe: Any,
    X: pd.DataFrame,
    y: pd.Series,
    avg: str,
    is_multiclass: bool,
) -> Tuple[float, float, float, float, float]:
    """
    Evalúa un solo fold y retorna (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
    """
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    cloned = clone(pipe)
    cloned.fit(X_tr, y_tr)
    y_pred = cloned.predict(X_val)
    
    acc = float(accuracy_score(y_val, y_pred))
    prec = float(precision_score(y_val, y_pred, average=avg, zero_division=0))
    rec = float(recall_score(y_val, y_pred, average=avg, zero_division=0))
    f1 = float(f1_score(y_val, y_pred, average=avg, zero_division=0))
    
    y_prob = cloned.predict_proba(X_val)
    if is_multiclass:
        auc = float(roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro"))
    else:
        auc = float(roc_auc_score(y_val, y_prob[:, 1]))
        
    return acc, prec, rec, f1, auc


@log_execution
def run_classification_cv(
    X: pd.DataFrame,
    y: pd.Series,
    pipelines: Dict[str, Pipeline],
    is_multiclass: bool = False,
    n_splits: int = CV_SPLITS,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta validación cruzada estratificada con ``StratifiedKFold`` para
    todos los pipelines, calculando Accuracy, Precision, Recall, F1-Score
    y ROC-AUC por fold de forma paralela.

    La **métrica principal** es ``f1_macro`` (NO accuracy).

    Args:
        X: DataFrame de características.
        y: Serie de etiquetas.
        pipelines: Diccionario de pipelines nombrados.
        is_multiclass: ``True`` si el target es multiclase.
        n_splits: Número de folds.
        seed: Semilla de aleatoriedad.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - ``df_formatted``: Tabla con ``mean ± std`` para presentación.
            - ``df_raw``: Tabla numérica con valores medios.
    """
    logger.info(
        f"Stratified {n_splits}-Fold CV (seed={seed}), "
        f"scoring principal: {SCORING_METRIC}"
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    raw_results: List[Dict[str, Any]] = []
    formatted_results: List[Dict[str, Any]] = []
    avg = "macro" if is_multiclass else "binary"

    from joblib import Parallel, delayed

    for name, pipe in pipelines.items():
        logger.info(f"Evaluando CV para: {name}...")
        try:
            results = Parallel(n_jobs=-1)(
                delayed(_evaluate_fold)(
                    train_idx, val_idx, pipe, X, y, avg, is_multiclass
                )
                for train_idx, val_idx in skf.split(X, y)
            )
            
            fold_metrics: Dict[str, List[float]] = {
                "Accuracy": [r[0] for r in results],
                "Precision": [r[1] for r in results],
                "Recall": [r[2] for r in results],
                "F1-Score": [r[3] for r in results],
                "ROC-AUC": [r[4] for r in results],
            }
        except Exception as e:
            logger.error(f"Error evaluando {name} en CV: {e}", exc_info=True)
            fold_metrics = {
                "Accuracy": [], "Precision": [], "Recall": [],
                "F1-Score": [], "ROC-AUC": [],
            }

        # Aggregate
        raw_row: Dict[str, Any] = {"Model": name}
        fmt_row: Dict[str, Any] = {"Model": name}
        for metric_name, values in fold_metrics.items():
            if values:
                m, s = float(np.mean(values)), float(np.std(values))
            else:
                m, s = 0.0, 0.0
            raw_row[f"{metric_name}_mean"] = m
            raw_row[f"{metric_name}_std"] = s
            fmt_row[metric_name] = f"{m:.4f} ± {s:.4f}"

        raw_results.append(raw_row)
        formatted_results.append(fmt_row)
        logger.info(
            f"[{name}] F1-{avg} Mean: "
            f"{raw_row.get('F1-Score_mean', 0):.4f}"
        )

    df_raw = pd.DataFrame(raw_results)
    df_formatted = pd.DataFrame(formatted_results)
    col_order = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_formatted = df_formatted[[c for c in col_order if c in df_formatted.columns]]
    return df_formatted, df_raw


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------

@log_execution
def evaluate_regressors(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    trained_regressors: Dict[str, Pipeline],
) -> pd.DataFrame:
    """
    Evalúa todos los regresores entrenados y calcula RMSE, MAE y R².

    Args:
        X_test: Datos de prueba.
        y_test: Variable objetivo real.
        trained_regressors: Regresores entrenados.

    Returns:
        pd.DataFrame: Tabla con métricas de regresión.
    """
    results: List[Dict[str, Any]] = []
    for name, pipe in trained_regressors.items():
        try:
            y_pred = pipe.predict(X_test)
            results.append({
                "Model": name,
                "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "MAE": float(mean_absolute_error(y_test, y_pred)),
                "R2": float(r2_score(y_test, y_pred)),
            })
            logger.info(f"[{name}] R²={results[-1]['R2']:.4f}")
        except Exception as e:
            logger.error(f"Error evaluando regresor {name}: {e}", exc_info=True)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_cv_results(
    df: pd.DataFrame,
    filename: str,
    output_dir: str = REPORTING_DIR,
) -> str:
    """
    Guarda los resultados de validación cruzada a un archivo CSV.

    Args:
        df: DataFrame a exportar.
        filename: Nombre del archivo (ej. ``binary_cv_results.csv``).
        output_dir: Directorio de destino.

    Returns:
        str: Ruta completa del archivo guardado.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    logger.info(f"Resultados guardados en: {path}")
    return path
