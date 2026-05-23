import pandas as pd
import logging

def validate_post_transformation(raw_df: pd.DataFrame, primary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validation pipeline: Compare before/after and verify schema.

    Compares the row counts of the raw and primary datasets, verifies that critical
    columns exist in the primary dataset, and checks for null values in those columns.

    Args:
        raw_df: The original raw DataFrame before any processing.
        primary_df: The fully processed and integrated primary DataFrame.

    Returns:
        pd.DataFrame: A single-row DataFrame containing validation metrics and status.
    """
    logger = logging.getLogger(__name__)
    
    # 1. Row Comparison
    rows_raw = len(raw_df)
    rows_primary = len(primary_df)
    
    # 2. Schema Validation (Critical Columns)
    critical_cols = ["EmpID", "Seniority_Years", "Total Training Cost"]
    schema_status = all(col in primary_df.columns for col in critical_cols)
    
    # 3. Null Check
    nulls_in_critical = {col: int(primary_df[col].isnull().sum()) for col in critical_cols if col in primary_df.columns}
    
    metrics = {
        "raw_count": rows_raw,
        "primary_count": rows_primary,
        "schema_valid": schema_status,
        "nulls_detected": sum(nulls_in_critical.values()),
        "status": "Validated" if (schema_status and sum(nulls_in_critical.values()) == 0) else "Issues Detected"
    }
    
    logger.info(f"Final Validation Report: {metrics}")
    return pd.DataFrame([metrics])
