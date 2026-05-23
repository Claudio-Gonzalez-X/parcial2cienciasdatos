"""
Módulo de aprendizaje no supervisado.

Implementa K-Means, DBSCAN, Agglomerative Clustering, PCA y GMM.
Calcula Silhouette, Davies-Bouldin, Inertia y Explained Variance Ratio.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from ev_parcial2_gonzalez.modeling.config import (
    AGGLOMERATIVE_N_CLUSTERS, DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
    DENDROGRAM_SAMPLE_SIZE, GMM_MAX_COMPONENTS, MAX_K_CLUSTERS,
    NUMERIC_FEATURES, PCA_VARIANCE_THRESHOLD_80, PCA_VARIANCE_THRESHOLD_90,
    RANDOM_STATE,
)
from ev_parcial2_gonzalez.modeling.utils import log_execution, setup_logger

logger = setup_logger(__name__)


@log_execution
def prepare_unsupervised_data(df: pd.DataFrame, features: List[str] | None = None) -> np.ndarray:
    """Imputa y escala features numéricas para clustering."""
    if features is None:
        features = NUMERIC_FEATURES
    X = df[features].copy()
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    logger.info(f"Datos no supervisados: {X.shape}")
    return X


@log_execution
def run_kmeans(X: np.ndarray, max_k: int = MAX_K_CLUSTERS) -> Dict:
    """K-Means con Inertia, Silhouette y Davies-Bouldin."""
    k_values = list(range(2, max_k + 1))
    inertias, silhouettes, dbs = [], [], []
    best_sil, best_km, best_labels = -1.0, None, None
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X, labels))
        silhouettes.append(sil)
        db = float(davies_bouldin_score(X, labels))
        dbs.append(db)
        logger.info(f"K={k} | Inertia={km.inertia_:.2f} | Sil={sil:.4f} | DB={db:.4f}")
        if sil > best_sil:
            best_sil, best_km, best_labels = sil, km, labels
    return {"k_values": k_values, "inertias": inertias, "silhouettes": silhouettes,
            "davies_bouldin": dbs, "best_kmeans": best_km, "best_labels": best_labels}


@log_execution
def run_dbscan(X: np.ndarray, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES) -> Dict:
    """DBSCAN para detección de clusters y outliers."""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_clusters = len(set(labels) - {-1})
    noise = int(np.sum(labels == -1))
    noise_pct = float(noise / len(X) * 100)
    logger.info(f"DBSCAN: {n_clusters} clusters, {noise} noise ({noise_pct:.1f}%)")
    return {"n_clusters": n_clusters, "noise_count": noise, "noise_pct": noise_pct, "labels": labels}


@log_execution
def run_agglomerative(X: np.ndarray, n_clusters: int = AGGLOMERATIVE_N_CLUSTERS) -> Dict:
    """Clustering jerárquico aglomerativo (Ward)."""
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X)
    np.random.seed(RANDOM_STATE)
    ss = min(DENDROGRAM_SAMPLE_SIZE, len(X))
    idx = np.random.choice(len(X), size=ss, replace=False)
    Z = linkage(X[idx], method="ward")
    logger.info(f"Agglomerative: {n_clusters} clusters, dendro sample={ss}")
    return {"labels": labels, "linkage_matrix": Z, "sample_size": ss}


@log_execution
def run_pca(X: np.ndarray) -> Dict:
    """PCA: varianza explicada, componentes para 80/90%, proyecciones 2D/3D."""
    pca = PCA(random_state=RANDOM_STATE)
    pca.fit(X)
    ev = pca.explained_variance_ratio_
    cev = np.cumsum(ev)
    c80 = int(np.argmax(cev >= PCA_VARIANCE_THRESHOLD_80) + 1)
    c90 = int(np.argmax(cev >= PCA_VARIANCE_THRESHOLD_90) + 1)
    X2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    X3d = PCA(n_components=3, random_state=RANDOM_STATE).fit_transform(X)
    logger.info(f"PCA: 80% var → {c80} comp, 90% → {c90} comp")
    return {"explained_variance": ev, "cumulative_variance": cev,
            "components_80": c80, "components_90": c90, "X_pca_2d": X2d, "X_pca_3d": X3d}


@log_execution
def run_gmm(X: np.ndarray, max_components: int = GMM_MAX_COMPONENTS) -> Dict:
    """GMM con AIC y BIC."""
    n_range = list(range(1, max_components + 1))
    bics, aics = [], []
    best_bic, best_labels = float("inf"), None
    for n in n_range:
        gmm = GaussianMixture(n_components=n, random_state=RANDOM_STATE)
        gmm.fit(X)
        b, a = float(gmm.bic(X)), float(gmm.aic(X))
        bics.append(b); aics.append(a)
        logger.info(f"GMM(n={n}) BIC={b:.2f} AIC={a:.2f}")
        if b < best_bic:
            best_bic, best_labels = b, gmm.predict(X)
    return {"n_components": n_range, "bics": bics, "aics": aics, "best_labels": best_labels}


def interpret_clusters(df: pd.DataFrame, labels: np.ndarray, features: List[str] | None = None) -> pd.DataFrame:
    """Tabla resumen de promedios por cluster para interpretación de negocio."""
    if features is None:
        features = NUMERIC_FEATURES
    d = df[features].copy()
    d["Cluster"] = labels
    return d.groupby("Cluster").mean()
