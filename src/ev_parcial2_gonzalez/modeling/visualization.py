"""
Módulo de visualizaciones para modelos supervisados.

Genera gráficos profesionales de matrices de confusión, curvas ROC,
curvas Precision-Recall, importancia de variables y comparaciones
de métricas entre modelos.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from ev_parcial2_gonzalez.modeling.config import RESULTS_PLOTS_DIR
from ev_parcial2_gonzalez.modeling.utils import log_execution, setup_logger

logger = setup_logger(__name__)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

@log_execution
def plot_confusion_matrix(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Genera y exporta una matriz de confusión para un modelo entrenado.

    Args:
        pipe: Pipeline entrenado.
        X_test: Datos de prueba.
        y_test: Etiquetas reales.
        title: Título del gráfico.
        filename: Nombre del archivo de salida.
        output_dir: Directorio de salida.

    Returns:
        str: Ruta completa del gráfico guardado.
    """
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ROC Curves
# ---------------------------------------------------------------------------

@log_execution
def plot_roc_curves(
    trained_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    is_multiclass: bool = False
) -> plt.Figure:
    """
    Genera curvas ROC comparativas para todos los modelos en un solo gráfico.

    Args:
        trained_models: Modelos entrenados.
        X_test: Datos de prueba.
        y_test: Etiquetas reales.
        is_multiclass: ``True`` para target multiclase (macro-average).
        filename: Nombre del archivo.
        output_dir: Directorio de salida.

    Returns:
        str: Ruta del gráfico.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, pipe in trained_models.items():
        try:
            if is_multiclass:
                classes = list(pipe.classes_)
                y_test_bin = label_binarize(y_test, classes=classes)
                n_classes = len(classes)
                y_prob = pipe.predict_proba(X_test)

                fpr_dict, tpr_dict = {}, {}
                for i in range(n_classes):
                    fpr_dict[i], tpr_dict[i], _ = roc_curve(
                        y_test_bin[:, i], y_prob[:, i]
                    )
                all_fpr = np.unique(
                    np.concatenate([fpr_dict[i] for i in range(n_classes)])
                )
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
                mean_tpr /= n_classes
                macro_auc = auc(all_fpr, mean_tpr)
                ax.plot(
                    all_fpr, mean_tpr,
                    label=f"{name} (Macro AUC={macro_auc:.3f})",
                    linewidth=2,
                )
            else:
                y_prob = pipe.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                ax.plot(
                    fpr, tpr,
                    label=f"{name} (AUC={roc_auc:.3f})",
                    linewidth=2,
                )
        except Exception as e:
            logger.warning(f"No se pudo calcular ROC para {name}: {e}")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    task_t = "Multiclass (Macro)" if is_multiclass else "Binary"
    ax.set_title(f"ROC Curve Comparison — {task_t}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Precision-Recall Curves
# ---------------------------------------------------------------------------

@log_execution
def plot_precision_recall_curves(
    trained_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> plt.Figure:
    """
    Genera curvas Precision-Recall comparativas (binary only).

    Args:
        trained_models: Modelos entrenados.
        X_test: Datos de prueba.
        y_test: Etiquetas reales (binarias).
        filename: Nombre del archivo.
        output_dir: Directorio de salida.

    Returns:
        str: Ruta del gráfico.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, pipe in trained_models.items():
        try:
            y_prob = pipe.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
            ax.plot(
                recall, precision,
                label=f"{name} (PR-AUC={pr_auc:.3f})",
                linewidth=2,
            )
        except Exception as e:
            logger.warning(f"PR curve fallida para {name}: {e}")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

@log_execution
def plot_feature_importance(
    pipe: Pipeline,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    title: str = "Feature Importance — Champion Model"
) -> plt.Figure:
    """
    Extrae y grafica la importancia de variables del modelo campeón.

    Funciona con modelos basados en árboles (``feature_importances_``).

    Args:
        pipe: Pipeline entrenado cuyo clasificador tenga
            ``feature_importances_``.
        feature_names: Nombres de las features tras preprocesamiento.
            Si es ``None`` se usan índices numéricos.
        top_n: Número máximo de features a mostrar.
        title: Título del gráfico.
        filename: Nombre del archivo.
        output_dir: Directorio de salida.

    Returns:
        str: Ruta del gráfico.
    """

    # Extract classifier from pipeline
    clf = pipe.named_steps.get("classifier") or pipe.steps[-1][1]
    if not hasattr(clf, "feature_importances_"):
        logger.warning("El modelo no tiene feature_importances_. Omitiendo.")
        return None

    importances = clf.feature_importances_

    # Try to get feature names from preprocessor
    if feature_names is None:
        try:
            preprocessor = pipe.named_steps.get("preprocessor")
            if preprocessor is not None:
                feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            pass
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    # Sort and truncate
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(
        range(len(top_features)),
        top_importances[::-1],
        color=sns.color_palette("viridis", len(top_features)),
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model Comparison Barplot
# ---------------------------------------------------------------------------

@log_execution
def plot_model_comparison(
    df_raw: pd.DataFrame,
    metric: str = "F1-Score_mean",
    title: str = "Model Comparison — F1-Score (macro)"
) -> plt.Figure:
    """
    Genera un gráfico de barras comparativo de métricas entre modelos.

    Args:
        df_raw: DataFrame crudo de resultados de CV (con columna ``Model``).
        metric: Columna de métrica a comparar.
        title: Título del gráfico.
        filename: Nombre del archivo.
        output_dir: Directorio de salida.

    Returns:
        str: Ruta del gráfico.
    """

    df_sorted = df_raw.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_sorted) * 0.5)))
    colors = sns.color_palette("coolwarm", len(df_sorted))
    bars = ax.barh(df_sorted["Model"], df_sorted[metric], color=colors)

    # Annotate
    for bar, val in zip(bars, df_sorted[metric]):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9,
        )

    ax.set_xlabel(metric.replace("_", " "), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig
