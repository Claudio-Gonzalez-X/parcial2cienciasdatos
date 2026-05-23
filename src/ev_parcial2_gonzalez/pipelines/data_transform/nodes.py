import pandas as pd
import numpy as np
from datetime import datetime

def merge_datasets(
    employee_df: pd.DataFrame,
    engagement_df: pd.DataFrame,
    recruitment_df: pd.DataFrame,
    training_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge datasets using Employee ID as primary key.

    Args:
        employee_df: DataFrame containing employee demographic data.
        engagement_df: DataFrame containing employee engagement survey data.
        recruitment_df: DataFrame containing recruitment data.
        training_df: DataFrame containing training and development data.

    Returns:
        pd.DataFrame: A single integrated DataFrame with all features merged on EmpID.
    """
    if "Employee ID" in engagement_df.columns:
        engagement_df = engagement_df.rename(columns={"Employee ID": "EmpID"})
    if "Employee ID" in training_df.columns:
        training_df = training_df.rename(columns={"Employee ID": "EmpID"})
        
    merged = pd.merge(employee_df, engagement_df, on="EmpID", how="left")
    
    # Advanced Transformation: Groupby Training Cost
    training_agg = training_df.groupby("EmpID")["Training Cost"].sum().reset_index()
    training_agg = training_agg.rename(columns={"Training Cost": "Total Training Cost"})
    
    merged = pd.merge(merged, training_agg, on="EmpID", how="left")
    return merged

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features and encode variables.

    Calculates employee seniority in years, normalizes the total training cost,
    and applies one-hot encoding for categorical variables like GenderCode.

    Args:
        df: Input DataFrame containing integrated employee data.

    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    df = df.copy()
    
    # Seniority
    if "StartDate" in df.columns:
        df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
        today = pd.Timestamp(datetime.now())
        df["Seniority_Years"] = (today - df["StartDate"]).dt.days / 365.25
        
    # Numeric Normalization (Example: min-max scaling for Total Training Cost)
    if "Total Training Cost" in df.columns:
        df["Total Training Cost"] = df["Total Training Cost"].fillna(0)
        max_val = df["Total Training Cost"].max()
        min_val = df["Total Training Cost"].min()
        if max_val != min_val:
            df["Normalized_Training_Cost"] = (df["Total Training Cost"] - min_val) / (max_val - min_val)
        else:
            df["Normalized_Training_Cost"] = 0
            
    # Categorical Encoding (Example: Dummy variables for GenderCode if exists)
    if "GenderCode" in df.columns:
        df = pd.get_dummies(df, columns=["GenderCode"], prefix="Gender", drop_first=True)
        
    return df
