# Informe Técnico - Parcial 2 Ciencia de Datos

## Resumen del Proyecto
Este proyecto implementa una arquitectura de datos robusta utilizando Kedro para procesar y analizar información de recursos humanos. Se han integrado múltiples fuentes de datos (empleados, reclutamiento, compromiso y capacitación) para generar una tabla maestra lista para análisis.

## Estructura de Pipelines
1. **data_ingestion**: Carga inicial de datos crudos.
2. **data_cleaning**: Limpieza, estandarización de fechas y normalización de textos.
3. **data_transform**: Integración de datasets mediante `EmpID` y cálculo de variables derivadas (antigüedad, costos).
4. **data_validation**: Verificación de integridad y generación de reportes de calidad.

## Calidad de Datos
El proceso de validación confirma:
- Cero pérdida de registros (3000 filas procesadas).
- Ausencia de nulos en columnas críticas (`EmpID`, `StartDate`).
- Estandarización completa de formatos.

## Conclusiones
La arquitectura permite una trazabilidad completa desde la capa `raw` hasta `primary`, asegurando que cualquier cambio en los parámetros de limpieza pueda ser replicado automáticamente.
