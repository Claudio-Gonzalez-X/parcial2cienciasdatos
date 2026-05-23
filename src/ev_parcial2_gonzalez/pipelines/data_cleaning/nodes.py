import pandas as pd
import numpy as np

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    return df.drop_duplicates()

def treat_outliers_iqr(df: pd.DataFrame, columns: list, iqr_multiplier: float) -> pd.DataFrame:
    """
    Treat outliers using the Interquartile Range (IQR) method.

    Args:
        df: Input DataFrame.
        columns: List of numeric columns to treat for outliers.
        iqr_multiplier: Multiplier for the IQR to determine bounds.

    Returns:
        DataFrame with outliers clipped to lower and upper bounds.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def impute_missing(df: pd.DataFrame, strategy: str, constant_value: str = "Desconocido") -> pd.DataFrame:
    """
    Impute missing values based on a specified strategy.

    Args:
        df: Input DataFrame.
        strategy: Numeric imputation strategy ('median' or 'mean').
        constant_value: Value to use for non-numeric column imputation.

    Returns:
        DataFrame with missing values filled.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(0)
            else:
                if strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(constant_value)
    return df

def standardize_dates(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    """
    Convert date columns to datetime objects, handling multiple formats.

    Args:
        df: Input DataFrame.
        date_cols: List of columns to convert.

    Returns:
        DataFrame with standardized datetime columns.
    """
    df = df.copy()
    for col in date_cols:
        if col in df.columns:
            # Added dayfirst=True to help with ambiguous formats like DD-MM-YYYY
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
    return df

def clean_strings(df: pd.DataFrame, str_cols: list) -> pd.DataFrame:
    """
    Clean string columns by stripping whitespace and title-casing.

    Args:
        df: Input DataFrame.
        str_cols: List of categorical columns to clean.

    Returns:
        DataFrame with normalized string values.
    """
    df = df.copy()
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    return df

def clean_data_node(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Combined cleaning node for the data_cleaning pipeline.

    Args:
        df: Raw DataFrame.
        params: Configuration dictionary with columns and strategies.

    Returns:
        Cleaned DataFrame ready for transformation.
    """
    df = remove_duplicates(df)
    df = standardize_dates(df, params["date_columns"])
    df = clean_strings(df, params["string_columns"])
    df = treat_outliers_iqr(df, params.get("outlier_columns", []), params.get("iqr_multiplier", 1.5))
    df = impute_missing(df, params["imputation"]["numeric_strategy"], params["imputation"]["constant_value"])
    return df
