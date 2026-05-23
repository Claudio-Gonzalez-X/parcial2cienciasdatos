"""
Módulo de preprocesamiento para Machine Learning.

Construye un ``ColumnTransformer`` robusto de scikit-learn que aplica
imputación, escalado estándar y codificación one-hot a las columnas
numéricas y categóricas respectivamente.
"""

from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ev_parcial2_gonzalez.modeling.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from ev_parcial2_gonzalez.modeling.utils import setup_logger

logger = setup_logger(__name__)


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construye un ``ColumnTransformer`` listo para integrarse en pipelines
    de ``imblearn`` o ``sklearn``.

    El sub-pipeline numérico imputa valores faltantes con la mediana y
    aplica ``StandardScaler``.  El sub-pipeline categórico imputa con
    ``'Unknown'`` y aplica ``OneHotEncoder`` con salida densa.

    Args:
        numeric_features: Lista de columnas numéricas.  Si es ``None``
            se usan las definidas en ``config.NUMERIC_FEATURES``.
        categorical_features: Lista de columnas categóricas.  Si es
            ``None`` se usan las definidas en ``config.CATEGORICAL_FEATURES``.

    Returns:
        ColumnTransformer: Preprocesador configurado.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    logger.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    logger.info(
        f"Categorical features ({len(categorical_features)}): {categorical_features}"
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    logger.info("Preprocessor construido exitosamente.")
    return preprocessor
