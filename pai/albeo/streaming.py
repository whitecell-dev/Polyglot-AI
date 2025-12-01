"""
Streaming Algebra - Time-windowed operations, debounce, rate
"""
import pandas as pd
import numpy as np
import duckdb
from typing import List, Dict, Any


def streaming_rolling(df: pd.DataFrame, value_col: str, window: int) -> pd.DataFrame:
    """
    Sliding window mean, min, max, sum.
    """
    df = df.copy()
    df[f"{value_col}_rolling_mean"] = df[value_col].rolling(window).mean()
    df[f"{value_col}_rolling_sum"] = df[value_col].rolling(window).sum()
    df[f"{value_col}_rolling_max"] = df[value_col].rolling(window).max()
    return df


def streaming_expanding(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Expanding / cumulative operations.
    """
    df = df.copy()
    df[f"{value_col}_cum_mean"] = df[value_col].expanding().mean()
    df[f"{value_col}_cum_sum"] = df[value_col].expanding().sum()
    return df


def streaming_duckdb_window(df: pd.DataFrame, time_col: str, value_col: str) -> pd.DataFrame:
    """
    Window functions using SQL semantics via DuckDB.
    """
    df = df.copy()
    con = duckdb.connect()
    # Ensure time column is treated as a known type for ordering
    df[time_col] = pd.to_datetime(df[time_col])
    con.register("t", df)

    out = con.execute(f"""
        SELECT *,
            AVG({value_col}) OVER (ORDER BY {time_col} ROWS 5 PRECEDING) AS rolling5_avg,
            SUM({value_col}) OVER (ORDER BY {time_col}) AS cumulative_sum
        FROM t
    """).df()

    return out
