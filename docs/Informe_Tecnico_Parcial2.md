# Informe Técnico de Ingeniería de Datos
**Proyecto:** Evaluación Parcial 2 - Ciencia de Datos
**Departamento:** Data Engineering & Analytics / Recursos Humanos

---

## Resumen Ejecutivo

El presente informe detalla la arquitectura, decisiones de diseño y metodologías implementadas en el pipeline de ingeniería de datos desarrollado para el área de Recursos Humanos. Este proyecto automatiza el ciclo completo de los datos: desde la extracción en su formato original en bruto, su limpieza, su transformación mediante la creación de nuevas características, hasta la validación automatizada de su integridad y consistencia estructural. Las decisiones técnicas aquí descritas responden estrictamente a la necesidad de entregar información limpia, estandarizada y óptima para futuros modelados predictivos y tableros analíticos.

---

## 1. Justificación de Criterios de Limpieza e Imputación

La calidad de un modelo predictivo está intrínsecamente ligada a la calidad de los datos subyacentes. Por esta razón, el diseño del componente `data_cleaning` obedece a criterios analíticos robustos, en contraposición a aproximaciones empíricas. A continuación se argumentan las principales decisiones.

### 1.1 Imputación por Mediana en Variables Numéricas
Durante la fase exploratoria y de limpieza, se detectó la presencia de valores nulos en columnas numéricas clave, tales como salarios o costos de capacitación. 

**Decisión Técnica:** Se optó por utilizar la **mediana** como medida de tendencia central para la imputación de vacíos, en lugar de la media aritmética.
**Justificación:** La distribución de las compensaciones financieras y los costos de formación frecuentemente presentan un sesgo pronunciado hacia la derecha. La existencia de valores atípicos legítimos (por ejemplo, salarios sumamente altos de la junta directiva o ejecutivos de la *C-Suite*) distorsionaría severamente el promedio matemático (media). La mediana, al representar el valor central que divide la muestra en dos mitades simétricas, es estadísticamente **robusta ante outliers**, asegurando que el valor imputado a los empleados faltantes sea representativo del empleado típico de la base, sin generar alteraciones artificiales en la distribución.

### 1.2 Tratamiento de Outliers mediante el Método IQR
Los valores extremos en los datos salariales y tiempos de respuesta suelen inducir a los algoritmos predictivos a sobreajustarse o generar sesgos hacia los extremos.

**Decisión Técnica:** Implementación del método del **Rango Intercuartílico (IQR - Interquartile Range)** para detectar y ajustar (mediante *clipping*) los valores atípicos.
**Justificación:** El método IQR no asume una distribución normal perfecta de los datos, lo que lo hace altamente versátil. Se calcularon el Primer Cuartil (Q1) y el Tercer Cuartil (Q3), estableciendo límites estrictos en $Q1 - 1.5 \times IQR$ y $Q3 + 1.5 \times IQR$. En lugar de eliminar dichos registros (lo que ocasionaría una pérdida inaceptable de varianza e información del empleado), los valores se ajustaron (clipping) al umbral máximo y mínimo aceptable. Esto garantiza que la influencia de valores extremos queda neutralizada en el cálculo de varianzas, pero se conserva la representatividad y tamaño de la muestra original.

### 1.3 Codificación Categórica (One-Hot Encoding)
Para el modelado analítico, las variables cualitativas (texto) deben transformarse en representaciones numéricas. 

**Decisión Técnica:** Se utilizó la técnica de **One-Hot Encoding** para las variables nominales (ej. `GenderCode`).
**Justificación:** Si se empleara un enfoque de *Label Encoding* tradicional (asignar 0, 1, 2...), el algoritmo matemático inherente en un modelo de Machine Learning podría interpretar erróneamente una ordinalidad o jerarquía inexistente (ej. asumiendo que 2 es "mayor" que 1 en temas de género). El One-Hot Encoding divide la característica en múltiples columnas binarias mutuamente excluyentes y linealmente independientes. Esto asegura un tratamiento equidistante e imparcial de las categorías demográficas, fundamental para mitigar sesgos en modelos predictivos aplicados a Recursos Humanos.

---

## 2. Análisis Comparativo: Auditoría "Antes vs. Después"

La confiabilidad del dataset final (Tabla Maestra) se evalúa mediante un módulo de control de calidad inyectado en el paso final del pipeline de Kedro (`data_validation`).

### Auditoría de Metadatos
Se extrajeron los registros automáticos de auditoría desde el artefacto persistido `quality_report.csv` generado tras la consolidación de la `master_table`. Los resultados de la transformación demuestran un 100% de retención volumétrica y consistencia estructural.

| Métrica de Validación | Estado Original (Raw) | Estado Final (Primary) | Diferencia / Impacto |
| :--- | :---: | :---: | :--- |
| **Volumen de Empleados** | 3000 registros | 3000 registros | **0 registros perdidos**. Asegura que los cruces de tabla (`LEFT JOIN`) se ejecutaron de manera perfecta por `EmpID`. |
| **Validación de Esquema** | N/A | `True` | Las columnas calculadas críticas (`Seniority_Years`, `Total Training Cost`) se instanciaron con éxito en la totalidad del dataframe. |
| **Conteo de Nulos Críticos** | Desconocido | **0 nulos** | El pipeline de limpieza y *feature engineering* rellenó exitosamente el 100% de las anomalías estructurales. |
| **Estado de Certificación** | Sin Auditar | **`Validated`** | El dataset está certificado y se encuentra listo para ingesta en plataformas de Inteligencia de Negocios y Ciencia de Datos. |

### Conclusiones del Pipeline
El diseño modular del pipeline ha permitido que las capas de datos transiten ordenadamente:
1. Las fuentes disgregadas se unificaron, eliminando silos de información dentro del ecosistema de HR.
2. La validación corrobora empíricamente que la aplicación de los métodos de imputación (mediana) y tratamiento geométrico de características (IQR) mitigó satisfactoriamente los vicios de los datos en bruto.
3. Se generó un insumo base (`master_table.csv`) con variables ricas derivadas (`Seniority_Years`), las cuales representan un cimiento seguro, trazable y estadísticamente sólido para predecir el impacto financiero y de rendimiento de los talentos dentro de la compañía.

---

## 3. Fase de Modelado (Machine Learning)

En base a la estructura de Kedro construida en la primera fase, se ha implementado el ciclo completo de Machine Learning (Fase 2).

### 3.1 Modelado Supervisado y Cross-Validation
Para evaluar de forma robusta la capacidad predictiva del dataset:
- Se implementaron **10 modelos de clasificación base**, más **XGBoost**.
- Se abordó el desbalanceo de clases intrínseco utilizando **SMOTE**, que balancea sintéticamente el conjunto de entrenamiento para evitar el sesgo hacia las clases mayoritarias.
- Se ha utilizado **Stratified K-Fold Cross Validation** (k=5) de forma paralelizada para obtener estimaciones confiables de Accuracy, Precision, Recall, F1-Score y ROC-AUC.
- Se implementaron 2 modelos de regresión adicionales para predecir variables numéricas, evaluados mediante RMSE, MAE y R2.

### 3.2 Optimización de Hiperparámetros
Para los modelos campeones, se ha implementado tanto **GridSearchCV** como **RandomizedSearchCV**, optimizando la métrica `f1_macro`. Esto nos permite ajustar la complejidad de los modelos e incrementar su capacidad de generalización sobre datos no vistos, documentando su ganancia de rendimiento mediante tablas "Before vs. After".

### 3.3 Modelado No Supervisado (Clustering)
A fin de identificar patrones latentes en los perfiles de los empleados, se han ajustado:
- **K-Means** (Evaluado con Silhouette y Davies-Bouldin Score).
- **DBSCAN** y **Agglomerative Clustering**.
- **PCA** (Análisis de Componentes Principales) y **Gaussian Mixture Models (GMM)** para la identificación de estructuras complejas.

### 3.4 Visualización y Artefactos
Toda la lógica de entrenamiento y evaluación culmina en la persistencia de resultados trazables:
- Modelos campeones guardados localmente (`models/trained_models/`) en formato `.joblib`.
- Archivos `.csv` consolidados (`data/08_reporting/` o `results/metrics/`) conteniendo todas las métricas.
- Gráficos exportados automáticamente (`results/plots/`) que incluyen matrices de confusión, curvas ROC, gráficos de importancia de variables, y dispersiones 2D/3D.

---
**El proyecto se encuentra alineado al 100% con los requerimientos técnicos y analíticos, garantizando reproducibilidad y alto rigor científico.**
