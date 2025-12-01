"""
Temporal Series - Lag/lead, rolling average, time-aligned joins
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def temporal_lag_lead(df: pd.DataFrame, value_col: str, lags: List[int], leads: List[int]) -> pd.DataFrame:
    """
    Add multiple lag & lead features.
    """
    df = df.copy()
    for lag in lags:
        df[f"{value_col}_lag{lag}"] = df[value_col].shift(lag)
    for lead in leads:
        df[f"{value_col}_lead{lead}"] = df[value_col].shift(-lead)
    return df


def temporal_resample(df: pd.DataFrame, ts_col: str, rule: str, agg_map: Dict[str, str]) -> pd.DataFrame:
    """
    Resample timeseries with arbitrary aggregation rules.
    """
    df = df.copy().set_index(ts_col)
    df.index = pd.to_datetime(df.index)
    return df.resample(rule).agg(agg_map).reset_index()


def temporal_join(df_left: pd.DataFrame, df_right: pd.DataFrame, on: str, tolerance: str) -> pd.DataFrame:
    """
    Time-aligned join with tolerance.
    e.g., match sensor readings within ±5 seconds.
    """
    left = df_left.copy()
    right = df_right.copy()
    left[on] = pd.to_datetime(left[on])
    right[on] = pd.to_datetime(right[on])

    return pd.merge_asof(
        left.sort_values(on),
        right.sort_values(on),
        on=on,
        tolerance=pd.Timedelta(tolerance),
        direction='nearest'
    )
