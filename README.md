# ev_parcial2_gonzalez — Pipeline de Machine Learning para Recursos Humanos

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org)

## Descripción

Proyecto de ciencia de datos que implementa un **pipeline completo de Machine Learning** para el departamento de Recursos Humanos, desde la ingeniería de datos hasta el modelado predictivo y la optimización de hiperparámetros.

### Componentes principales

1. **Ingeniería de Datos (Kedro)**: Pipelines modulares de ingestión, limpieza, transformación y validación.
2. **Modelado Supervisado**: 10 clasificadores + XGBoost con SMOTE, 2 regresores, validación cruzada estratificada.
3. **Modelado No Supervisado**: K-Means, DBSCAN, Agglomerative, PCA, GMM con interpretación de clusters.
4. **Optimización**: GridSearchCV / RandomizedSearchCV con `scoring='f1_macro'`.
5. **Visualización**: Curvas ROC, PR, matrices de confusión, importancia de variables, dendrogramas, PCA 2D/3D.

## Estructura del Proyecto

```
ev-parcial2-gonzalez/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb      # EDA
│   ├── 02_supervised_modeling.ipynb        # Entrenamiento supervisado
│   ├── 03_unsupervised_modeling.ipynb      # Clustering y PCA
│   ├── 04_model_evaluation.ipynb          # CV y métricas comparativas
│   ├── 05_hyperparameter_optimization.ipynb # Tuning
│   └── 06_final_analysis.ipynb            # Integración final
│
├── src/ev_parcial2_gonzalez/
│   ├── pipelines/                         # Pipelines Kedro (data engineering)
│   │   ├── data_ingestion/
│   │   ├── data_cleaning/
│   │   ├── data_transform/
│   │   └── data_validation/
│   └── modeling/                          # Módulos de ML
│       ├── config.py                      # Configuración centralizada
│       ├── utils.py                       # Logger y serialización
│       ├── preprocessing.py               # ColumnTransformer
│       ├── model_training.py              # 10 clasificadores + regresores
│       ├── model_evaluation.py            # CV estratificada + métricas
│       ├── hyperparameter_tuning.py       # Grid/RandomizedSearchCV
│       ├── unsupervised.py                # 5 modelos no supervisados
│       ├── visualization.py               # Gráficos supervisados
│       ├── unsupervised_visualization.py  # Gráficos no supervisados
│       └── train.py                       # Orquestador principal
│
├── data/
│   ├── 01_raw/                            # Datos originales
│   ├── 02_intermediate/                   # Datos limpios
│   ├── 03_primary/master_table.csv        # Tabla maestra
│   └── 08_reporting/                      # CSVs de resultados
│
├── models/trained_models/                 # Modelos serializados (.pkl)
├── results/
│   ├── plots/                             # Visualizaciones PNG
│   ├── metrics/                           # Métricas adicionales
│   └── reports/                           # Reportes generados
│
├── docs/                                  # Informe técnico
├── conf/base/                             # Configuración Kedro
├── tests/                                 # Tests unitarios
├── requirements.txt                       # Dependencias
└── README.md
```

## Instalación

```bash
# 1. Crear y activar entorno virtual
python -m venv kedro-env
kedro-env\Scripts\activate   # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Instalar proyecto en modo editable
pip install -e .
```

## Dependencias Principales

| Paquete | Versión Mínima |
| :--- | :--- |
| pandas | ≥ 2.2 |
| numpy | ≥ 1.26 |
| scikit-learn | ≥ 1.5 |
| imbalanced-learn | ≥ 0.12 |
| matplotlib | ≥ 3.9 |
| seaborn | ≥ 0.13 |
| scipy | ≥ 1.13 |
| joblib | ≥ 1.4 |
| xgboost | ≥ 2.1 |
| kedro | ≥ 0.19 |

## Ejecución

### Pipeline de Ingeniería de Datos (Kedro)
```bash
kedro run
```

### Pipeline Completo de Machine Learning
```bash
cd ev-parcial2-gonzalez
kedro-env\Scripts\python -m ev_parcial2_gonzalez.modeling.train
```

### Notebooks Interactivos
```bash
jupyter notebook notebooks/
```

## Modelos Implementados

### Supervisados (Clasificación)
| # | Modelo | Pipeline |
| :--- | :--- | :--- |
| 1 | Logistic Regression | SMOTE + StandardScaler |
| 2 | Decision Tree | SMOTE + StandardScaler |
| 3 | Random Forest | SMOTE + StandardScaler |
| 4 | Gradient Boosting | SMOTE + StandardScaler |
| 5 | SVC | SMOTE + StandardScaler |
| 6 | KNN | SMOTE + StandardScaler |
| 7 | Gaussian NB | SMOTE + StandardScaler |
| 8 | Extra Trees | SMOTE + StandardScaler |
| 9 | AdaBoost | SMOTE + StandardScaler |
| 10 | MLP Neural Network | SMOTE + StandardScaler |
| 11 | XGBoost (extra) | SMOTE + StandardScaler |

### Supervisados (Regresión)
| # | Modelo |
| :--- | :--- |
| 1 | Linear Regression |
| 2 | Random Forest Regressor |

### No Supervisados
| # | Modelo | Métricas |
| :--- | :--- | :--- |
| 1 | K-Means | Inertia, Silhouette, Davies-Bouldin |
| 2 | DBSCAN | Clusters, Noise % |
| 3 | Agglomerative | Ward Linkage, Dendrogram |
| 4 | PCA | Explained Variance, Scree Plot |
| 5 | GMM | AIC, BIC |

## Resultados Generados

- `data/08_reporting/binary_cv_results.csv` — CV binaria
- `data/08_reporting/multiclass_cv_results.csv` — CV multiclase
- `data/08_reporting/regression_results.csv` — Métricas de regresión
- `data/08_reporting/tuning_comparison.csv` — Before vs After tuning
- `results/plots/` — Todas las visualizaciones (ROC, PR, CM, Feature Importance, Elbow, PCA, etc.)
- `models/trained_models/` — Modelos serializados (.pkl)

## Reproducibilidad

- `RANDOM_STATE = 42` en todas las operaciones estocásticas.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Scoring principal: `f1_macro`.
- Datos versionados en carpetas Kedro (01_raw → 03_primary → 08_reporting).
