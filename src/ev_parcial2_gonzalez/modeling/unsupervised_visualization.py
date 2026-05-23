"""
Visualizaciones para modelos no supervisados.
Retorna siempre fig — nunca llama plt.show() ni plt.close().
El caller (train.py o notebook) decide qué hacer con la figura.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples

from ev_parcial2_gonzalez.modeling.utils import log_execution, setup_logger

logger = setup_logger(__name__)


@log_execution
def plot_elbow_silhouette(k_values, inertias, silhouettes):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(k_values, inertias, "o-", lw=2, color="tab:blue", label="Inertia")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, ls="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(k_values, silhouettes, "s-", lw=2, color="tab:orange", label="Silhouette")
    ax2.set_ylabel("Silhouette", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title("KMeans — Elbow & Silhouette", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig  # caller guarda/muestra


@log_execution
def plot_dendrogram(linkage_matrix, sample_size):
    fig, ax = plt.subplots(figsize=(12, 6))

    dendrogram(
        linkage_matrix,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.7 * max(linkage_matrix[:, 2]),
        ax=ax,
    )

    ax.set_title(f"Dendrograma Agglomerative (muestra={sample_size})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Muestras")
    ax.set_ylabel("Distancia")
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    fig.tight_layout()
    return fig


@log_execution
def plot_pca_2d(X_2d, labels, title="PCA 2D"):
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=labels, cmap="viridis", alpha=0.7, s=20)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    return fig


@log_execution
def plot_pca_3d(X_3d, labels, title="PCA 3D"):
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                         c=labels, cmap="viridis", alpha=0.7, s=20)
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title, fontsize=13, fontweight="bold")

    # tight_layout rompe plots 3D — usar subplots_adjust
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
    return fig


@log_execution
def plot_silhouette(X_scaled, labels):
    sil_vals  = silhouette_samples(X_scaled, labels)
    n_clusters = len(set(labels) - {-1})

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for i in range(n_clusters):
        vals = np.sort(sil_vals[labels == i])
        y_upper = y_lower + vals.shape[0]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * vals.shape[0], str(i), fontsize=9)
        y_lower = y_upper + 10

    avg = np.mean(sil_vals)
    ax.axvline(avg, color="red", ls="--", lw=2, label=f"Promedio={avg:.3f}")
    ax.set_xlabel("Coeficiente Silhouette")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Plot", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


@log_execution
def plot_cluster_heatmap(cluster_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cluster_summary, annot=True, fmt=".2f",
                cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Perfiles de Cluster — Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


@log_execution
def plot_gmm_aic_bic(n_components, bics, aics):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_components, bics, "o-", lw=2, label="BIC")
    ax.plot(n_components, aics, "s-", lw=2, label="AIC")
    ax.set_xlabel("Componentes")
    ax.set_ylabel("Score")
    ax.set_title("GMM — AIC / BIC", fontsize=13, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig