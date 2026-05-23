import pandas as pd
import io

def generate_diagnostic_report(
    employee_df: pd.DataFrame,
    engagement_df: pd.DataFrame,
    recruitment_df: pd.DataFrame,
    training_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate an initial diagnostic report for the raw datasets.

    Args:
        employee_df: Raw employee data.
        engagement_df: Raw engagement survey data.
        recruitment_df: Raw recruitment data.
        training_df: Raw training and development data.

    Returns:
        DataFrame containing shape and null count metrics for each dataset.
    """
    datasets = {
        "employee_data": employee_df,
        "engagement_data": engagement_df,
        "recruitment_data": recruitment_df,
        "training_data": training_df
    }
    
    report_data = []
    for name, df in datasets.items():
        metrics = {
            "dataset": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "null_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "columns_list": ", ".join(df.columns.tolist())
        }
        report_data.append(metrics)
        
    return pd.DataFrame(report_data)
