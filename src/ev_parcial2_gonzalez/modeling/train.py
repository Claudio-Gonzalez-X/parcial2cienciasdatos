"""
Orquestador principal del pipeline ML.
Versión simplificada y estable para Kedro + Jupyter.
"""

import os
import sys

# ── Backend matplotlib SOLO si NO estamos en Jupyter ──────────────
# Debe ir antes de cualquier import que toque matplotlib
if "ipykernel" not in sys.modules:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ev_parcial2_gonzalez.modeling.config import (
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PRIMARY_DATA_PATH,
    RANDOM_STATE,
    REPORTING_DIR,
    RESULTS_PLOTS_DIR,
    TEST_SIZE,
)

from ev_parcial2_gonzalez.modeling.utils import (
    log_execution,
    save_model,
    setup_logger,
)

from ev_parcial2_gonzalez.modeling.preprocessing import build_preprocessor

from ev_parcial2_gonzalez.modeling.model_training import (
    build_classification_pipeline,
    get_supervised_classifiers,
    train_all_regressors,
)

from ev_parcial2_gonzalez.modeling.model_evaluation import (
    evaluate_regressors,
    run_classification_cv,
    save_cv_results,
)

from ev_parcial2_gonzalez.modeling.hyperparameter_tuning import (
    compare_before_after,
    get_param_grid,
    run_randomized_search,
)

from ev_parcial2_gonzalez.modeling.unsupervised import (
    interpret_clusters,
    prepare_unsupervised_data,
    run_agglomerative,
    run_gmm,
    run_kmeans,
    run_pca,
)

from ev_parcial2_gonzalez.modeling.unsupervised_visualization import (
    plot_cluster_heatmap,
    plot_dendrogram,
    plot_elbow_silhouette,
    plot_gmm_aic_bic,
    plot_pca_2d,
    plot_pca_3d,
    plot_silhouette,
)

from ev_parcial2_gonzalez.modeling.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
)

logger = setup_logger(__name__)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def save_figure(fig, filename):
    """Guarda figura y la cierra. Nunca cierra antes de guardar."""
    if fig is None:
        return
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)   # DESPUÉS de savefig, nunca antes
    logger.info(f"Guardado: {path}")


# -------------------------------------------------------------------
# Data
# -------------------------------------------------------------------

def load_data(filepath=PRIMARY_DATA_PATH):
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    df = pd.read_csv(filepath)
    logger.info(f"Dataset cargado: {df.shape}")
    return df


def prepare_targets(df):
    status = df["EmployeeStatus"].astype(str).str.strip()
    perf   = df["Performance Score"].astype(str).str.strip()

    y_bin = status.isin([
        "Voluntarily Terminated",
        "Terminated For Cause"
    ]).astype(int)

    y_mul = perf
    X     = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    return X, y_bin, y_mul


# -------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------

@log_execution
def run_full_pipeline(filepath=PRIMARY_DATA_PATH):

    for d in [MODELS_DIR, RESULTS_PLOTS_DIR, REPORTING_DIR]:
        os.makedirs(d, exist_ok=True)

    # 1. Load
    df = load_data(filepath)
    X, y_bin, y_mul = prepare_targets(df)

    # 2. Encode multiclass
    le    = LabelEncoder()
    y_mul = pd.Series(le.fit_transform(y_mul), index=y_mul.index)

    # 3. Split
    X_train, X_test, yb_train, yb_test = train_test_split(
        X, y_bin, test_size=TEST_SIZE, stratify=y_bin, random_state=RANDOM_STATE
    )
    _, _, ym_train, ym_test = train_test_split(
        X, y_mul, test_size=TEST_SIZE, stratify=y_mul, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()

    # ── SUPERVISED ──────────────────────────────────────────────────
    classifiers = {
        n: build_classification_pipeline(preprocessor, c)
        for n, c in get_supervised_classifiers().items()
    }

    fmt_bin, raw_bin = run_classification_cv(X, y_bin, classifiers, is_multiclass=False)
    save_cv_results(raw_bin, "binary_cv_results.csv")

    fmt_mul, raw_mul = run_classification_cv(X, y_mul, classifiers, is_multiclass=True)
    save_cv_results(raw_mul, "multiclass_cv_results.csv")

    trained_bin = {n: clone(p).fit(X_train, yb_train) for n, p in classifiers.items()}
    trained_mul = {n: clone(p).fit(X_train, ym_train) for n, p in classifiers.items()}

    # ── REGRESSION ──────────────────────────────────────────────────
    y_reg           = df["Current Employee Rating"].astype(float)
    X_reg           = X.drop(columns=["Current Employee Rating"], errors="ignore")
    num_features_reg = [f for f in NUMERIC_FEATURES if f != "Current Employee Rating"]
    preprocessor_reg = build_preprocessor(numeric_features=num_features_reg)

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    trained_reg = train_all_regressors(Xr_train, yr_train, preprocessor_reg)
    reg_results = evaluate_regressors(Xr_test, yr_test, trained_reg)
    save_cv_results(reg_results, "regression_results.csv")

    # ── TUNING ──────────────────────────────────────────────────────
    champ_name = raw_bin.sort_values("F1-Score_mean", ascending=False).iloc[0]["Model"]

    try:
        rs = run_randomized_search(
            classifiers[champ_name], X_train, yb_train,
            get_param_grid(champ_name), n_iter=10
        )
        comparison = compare_before_after(raw_bin, champ_name, rs.best_score_)
        save_cv_results(comparison, "tuning_comparison.csv")
        save_model(rs.best_estimator_, os.path.join(MODELS_DIR, f"{champ_name}_tuned.joblib"))
    except Exception as e:
        logger.error(f"Tuning error: {e}")

    # ── SAVE MODELS ─────────────────────────────────────────────────
    for n, p in trained_bin.items():
        save_model(p, os.path.join(MODELS_DIR, f"{n}_binary.joblib"))
    for n, p in trained_mul.items():
        save_model(p, os.path.join(MODELS_DIR, f"{n}_multiclass.joblib"))
    for n, p in trained_reg.items():
        save_model(p, os.path.join(MODELS_DIR, f"{n}_regressor.joblib"))

    # ── VISUALIZATIONS ──────────────────────────────────────────────
    best_model = trained_bin[champ_name]

    save_figure(
        plot_confusion_matrix(best_model, X_test, yb_test,
                              title=f"Confusion Matrix - {champ_name}"),
        "confusion_matrix.png"
    )
    save_figure(
        plot_feature_importance(best_model, title=f"Feature Importance - {champ_name}"),
        "feature_importance.png"
    )
    save_figure(
        plot_roc_curves(trained_bin, X_test, yb_test),
        "roc_curve.png"
    )
    save_figure(
        plot_roc_curves(trained_mul, X_test, ym_test, is_multiclass=True),
        "multiclass_roc_curve.png"
    )
    save_figure(
        plot_precision_recall_curves(trained_bin, X_test, yb_test),
        "precision_recall_curve.png"
    )
    save_figure(
        plot_model_comparison(raw_bin, metric="F1-Score_mean", title="Binary Model Comparison"),
        "model_comparison.png"
    )
    save_figure(
        plot_model_comparison(raw_mul, metric="F1-Score_mean", title="Multiclass Model Comparison"),
        "multiclass_model_comparison.png"
    )

    # ── UNSUPERVISED ─────────────────────────────────────────────────
    X_scaled = prepare_unsupervised_data(df)
    km  = run_kmeans(X_scaled)
    agg = run_agglomerative(X_scaled)
    pca = run_pca(X_scaled)
    gmm = run_gmm(X_scaled)

    save_figure(plot_elbow_silhouette(km["k_values"], km["inertias"], km["silhouettes"]), "elbow_plot.png")
    save_figure(plot_silhouette(X_scaled, km["best_labels"]), "silhouette_plot.png")
    save_figure(plot_dendrogram(agg["linkage_matrix"], agg["sample_size"]), "dendrogram.png")
    save_figure(plot_pca_2d(pca["X_pca_2d"], km["best_labels"], title="PCA 2D"), "pca_2d.png")
    save_figure(plot_pca_3d(pca["X_pca_3d"], km["best_labels"], title="PCA 3D"), "pca_3d.png")
    save_figure(plot_gmm_aic_bic(gmm["n_components"], gmm["bics"], gmm["aics"]), "gmm_aic_bic.png")

    cluster_summary = interpret_clusters(df, km["best_labels"])
    save_figure(plot_cluster_heatmap(cluster_summary), "cluster_heatmap.png")
    cluster_summary.to_csv(os.path.join(REPORTING_DIR, "cluster_profiles.csv"))

    logger.info("PIPELINE COMPLETADO")


if __name__ == "__main__":
    run_full_pipeline()