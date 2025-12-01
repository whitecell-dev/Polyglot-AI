"""
Python Bridge - Structural data operations for IMPO ↔ ALBEO communication

ENFORCES:
- Returns only primitives (int, str, bool, dict, list)
- Never returns complex Pandas/NumPy objects
- Provides defensive operations only (no mutations)
"""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union
import json


class PandasStructuralBridge:
    """
    BRIDGE LAYER: Safe boundary for Pandas ↔ IMPO data transfer
    """
    
    @staticmethod
    def shape_info(df: pd.DataFrame) -> Dict[str, int]:
        """
        SEMANTIC MAPPING: Python tuple → primitive dict
        
        Returns: {'rows': int, 'cols': int} - primitives only
        """
        rows, cols = df.shape
        return {
            'rows': int(rows),  # Explicit primitive conversion
            'cols': int(cols)   # Never return NumPy int64
        }
    
    @staticmethod
    def schema_validation(df: pd.DataFrame, expected_columns: List[str]) -> Dict[str, Any]:
        """
        SEMANTIC MAPPING: DataFrame schema → primitive validation result
        """
        actual = list(df.columns)
        missing = [c for c in expected_columns if c not in actual]
        extra = [c for c in actual if c not in expected_columns]
        
        return {
            'is_valid': bool(len(missing) == 0),
            'missing_columns': [str(c) for c in missing],
            'extra_columns': [str(c) for c in extra],
            'expected_count': int(len(expected_columns)),
            'actual_count': int(len(actual)),
            'column_names': [str(c) for c in actual]
        }
    
    @staticmethod
    def basic_info(df: pd.DataFrame) -> Dict[str, Any]:
        """
        SEMANTIC MAPPING: DataFrame metadata → Primitives
        """
        return {
            'row_count': int(len(df)),
            'column_names': [str(c) for c in df.columns],
            'is_empty': bool(df.empty),
            'memory_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
    
    @staticmethod
    def get_column_values(df: pd.DataFrame, col_name: str) -> List[Any]:
        """
        SEMANTIC MAPPING: Pandas Series → Python list
        
        CRITICAL: Returns list, not Series, to avoid complex object leakage
        """
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found")
        
        # Convert to list, handling various pandas types
        values = df[col_name].tolist()
        
        # Ensure primitives (convert numpy types)
        return [
            float(v) if isinstance(v, (np.integer, np.floating)) else v
            for v in values
        ]
    
    @staticmethod
    def json_lens_extract(df: pd.DataFrame, column: str, path: List[str]) -> Dict[str, List[Any]]:
        """
        Extract nested JSON fields using a composable lens path.
        Returns as primitive dict with list of extracted values.
        """
        def lens(obj):
            if not obj:
                return None
            try:
                if isinstance(obj, str):
                    obj = json.loads(obj)
                
                for key in path:
                    if not isinstance(obj, dict) or key not in obj:
                        return None
                    obj = obj[key]
                return obj
            except:
                return None
        
        extracted = df[column].apply(lens).tolist()
        return {
            'extracted_values': extracted,
            'path': path,
            'count': len(extracted)
        }
    
    @staticmethod
    def extract_numerical_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Extract numerical statistics as primitives
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Column '{column}' is not numeric")
        
        return {
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': float(series.mean()),
            'std': float(series.std()),
            'count': int(len(series)),
            'null_count': int(series.isnull().sum())
        }
    
    @staticmethod
    def describe_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive DataFrame description as primitives
        """
        return {
            'shape': PandasStructuralBridge.shape_info(df),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'memory_usage': int(df.memory_usage(deep=True).sum()),
            'null_counts': {col: int(df[col].isnull().sum()) for col in df.columns},
            'unique_counts': {col: int(df[col].nunique()) for col in df.columns}
        }
