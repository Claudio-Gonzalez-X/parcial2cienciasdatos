"""
Configuración global centralizada para el módulo de modelado.

Define todas las constantes, semillas de aleatoriedad, rutas de salida y
listas de features utilizadas a lo largo del pipeline de Machine Learning.
Centralizar estos valores garantiza reproducibilidad absoluta y facilita
el ajuste de parámetros sin modificar el código fuente.
"""

from typing import List

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Validación cruzada
# ---------------------------------------------------------------------------
CV_SPLITS: int = 5
SCORING_METRIC: str = "f1_macro"

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
N_JOBS: int = -1
SMOTE_SAMPLING_STRATEGY: str = "auto"

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: List[str] = [
    "Engagement Score",
    "Satisfaction Score",
    "Work-Life Balance Score",
    "Total Training Cost",
    "Seniority_Years",
    "Current Employee Rating",
]

CATEGORICAL_FEATURES: List[str] = [
    "Title",
    "BusinessUnit",
    "EmployeeType",
    "PayZone",
    "EmployeeClassificationType",
    "DepartmentType",
    "Division",
    "State",
    "LocationCode",
    "RaceDesc",
    "MaritalDesc",
    "Gender_Male",
]

# ---------------------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------------------
MODELS_DIR: str = "models/trained_models"
RESULTS_PLOTS_DIR: str = "results/plots"
RESULTS_METRICS_DIR: str = "results/metrics"
RESULTS_REPORTS_DIR: str = "results/reports"
REPORTING_DIR: str = "data/08_reporting"
PRIMARY_DATA_PATH: str = "data/03_primary/master_table.csv"

# ---------------------------------------------------------------------------
# Unsupervised
# ---------------------------------------------------------------------------
MAX_K_CLUSTERS: int = 8
DBSCAN_EPS: float = 1.5
DBSCAN_MIN_SAMPLES: int = 5
PCA_VARIANCE_THRESHOLD_80: float = 0.80
PCA_VARIANCE_THRESHOLD_90: float = 0.90
GMM_MAX_COMPONENTS: int = 8
AGGLOMERATIVE_N_CLUSTERS: int = 3
DENDROGRAM_SAMPLE_SIZE: int = 150
