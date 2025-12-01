"""
Scoring functions - ML feature engineering, model scoring, etc.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common financial ratios.
    """
    df = df.copy()
    
    if 'income' in df.columns and 'debt' in df.columns:
        df['dti_ratio'] = (df['debt'] / df['income']) * 100
    
    if 'loan_amount' in df.columns and 'income' in df.columns:
        df['loan_to_income'] = (df['loan_amount'] / df['income']) * 100
    
    if 'credit_score' in df.columns:
        df['meets_credit_threshold'] = df['credit_score'] >= 620
    
    return df


def calculate_aggregate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate aggregate metrics and return as primitives.
    """
    metrics = {}
    
    if 'dti_ratio' in df.columns:
        metrics['avg_dti'] = float(df['dti_ratio'].mean())
        metrics['max_dti'] = float(df['dti_ratio'].max())
    
    if 'credit_score' in df.columns:
        metrics['avg_credit'] = float(df['credit_score'].mean())
        metrics['min_credit'] = float(df['credit_score'].min())
    
    if 'meets_credit_threshold' in df.columns:
        metrics['meets_credit_count'] = int(df['meets_credit_threshold'].sum())
    
    return metrics


def ml_feature_engineering(df: pd.DataFrame, 
                          value_columns: List[str] = None) -> pd.DataFrame:
    """
    Create ML features from base columns.
    """
    df = df.copy()
    
    if value_columns is None:
        # Auto-detect numeric columns
        value_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in value_columns:
        if col in df.columns:
            # Basic statistical features
            df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
            df[f'{col}_log'] = np.log1p(df[col].abs())
            
            # Relative features
            if len(value_columns) > 1:
                for other in value_columns:
                    if other != col and other in df.columns:
                        df[f'{col}_to_{other}'] = df[col] / (df[other] + 1e-10)
    
    return df
