"""
Dataflow Composition - split, merge, tee, failover, all declarative
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Tuple


def dataflow_split(df: pd.DataFrame, parts: int) -> List[pd.DataFrame]:
    """
    Split DataFrame into N nearly equal partitions.
    """
    df = df.copy()
    return np.array_split(df, parts)


def dataflow_merge(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple DataFrames vertically.
    """
    dfs_copy = [d.copy() for d in dfs]
    return pd.concat(dfs_copy, ignore_index=True)


def dataflow_tee(df: pd.DataFrame, funcs: List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]) -> Dict[str, pd.DataFrame]:
    """
    Apply multiple operations to same source (like UNIX tee).
    funcs is list of (name, fn)
    """
    df = df.copy()
    out = {}
    for name, fn in funcs:
        out[name] = fn(df)
    return out


def dataflow_failover(df: pd.DataFrame, funcs: List[Callable[[pd.DataFrame], pd.DataFrame]]) -> pd.DataFrame:
    """
    Try each function in order until one succeeds.
    """
    df = df.copy()
    for fn in funcs:
        try:
            return fn(df)
        except Exception:
            continue
    raise RuntimeError("All failover paths failed")
