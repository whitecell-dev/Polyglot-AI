"""
Lenses & Traversals - Deep JSON editing via composable paths
"""
import json
import pandas as pd
from typing import List


def json_lens_extract(df: pd.DataFrame, column: str, path: List[str]) -> pd.DataFrame:
    """
    Extract nested JSON fields using a composable lens path.
    Example path: ['a','b','c'] extracts obj['a']['b']['c']
    """
    df = df.copy()

    def lens(obj):
        if not obj:
            return None
        try:
            # Handle string representations of JSON
            if isinstance(obj, str):
                obj = json.loads(obj)

            for key in path:
                if not isinstance(obj, dict) or key not in obj:
                    return None
                obj = obj[key]
            return obj
        except:
            return None

    df[f"json__{'__'.join(path)}"] = df[column].apply(lens)
    return df


def json_normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize nested JSON objects into flat columns.
    """
    df = df.copy()

    # Try to parse stringified JSON first
    def safe_parse(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except:
                return None
        return x

    df[column] = df[column].apply(safe_parse)

    # Filter out rows where the column contains non-dict/list values for normalization
    valid_data = df[df[column].apply(lambda x: isinstance(x, (dict, list)) and bool(x))].copy()

    if valid_data.empty:
        # Return original df unchanged if no valid JSON
        return df

    # Normalize the valid data
    expanded = pd.json_normalize(valid_data[column])
    expanded.columns = [f"{column}.{c}" for c in expanded.columns]

    # Reindex expanded data to align with original valid data index
    expanded.index = valid_data.index

    # Combine original data (dropping the complex column) with the new expanded columns
    result_df = df.drop(columns=[column]).join(expanded, how='left')
    return result_df
