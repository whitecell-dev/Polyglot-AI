"""
Schema Bridge - Type validation and schema enforcement
"""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union, Type, get_type_hints
from dataclasses import is_dataclass, asdict
import inspect


class SchemaValidator:
    """
    Validates schemas and types across bridge boundaries
    """
    
    @staticmethod
    def validate_dataframe_schema(df: pd.DataFrame, schema: Dict[str, Type]) -> Dict[str, Any]:
        """
        Validate DataFrame against expected schema
        Returns validation result as primitives
        """
        results = {}
        violations = []
        
        for col_name, expected_type in schema.items():
            if col_name not in df.columns:
                violations.append(f"Missing column: {col_name}")
                continue
            
            col_type = df[col_name].dtype
            actual_type = str(col_type)
            
            # Type checking logic
            is_valid = SchemaValidator._check_type_compatibility(col_type, expected_type)
            
            results[col_name] = {
                'expected': str(expected_type),
                'actual': actual_type,
                'valid': is_valid,
                'sample_value': str(df[col_name].iloc[0]) if len(df) > 0 else None
            }
            
            if not is_valid:
                violations.append(f"Type mismatch for {col_name}: expected {expected_type}, got {actual_type}")
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'column_results': results,
            'total_columns': len(schema),
            'valid_columns': sum(1 for r in results.values() if r['valid'])
        }
    
    @staticmethod
    def _check_type_compatibility(actual_type: Any, expected_type: Type) -> bool:
        """
        Check if actual pandas/numpy type is compatible with expected Python type
        """
        # Map pandas/numpy types to Python types
        type_mapping = {
            'int64': int, 'int32': int, 'int16': int, 'int8': int,
            'float64': float, 'float32': float, 'float16': float,
            'bool': bool,
            'object': str, 'string': str,
            'datetime64[ns]': 'datetime',
            'category': 'category'
        }
        
        actual_str = str(actual_type)
        mapped_type = type_mapping.get(actual_str, None)
        
        if mapped_type is None:
            return False
        
        # Check compatibility
        if expected_type == Any:
            return True
        elif expected_type == str and mapped_type == str:
            return True
        elif expected_type == int and mapped_type == int:
            return True
        elif expected_type == float and mapped_type == float:
            return True
        elif expected_type == bool and mapped_type == bool:
            return True
        
        return False
    
    @staticmethod
    def validate_function_signature(func: callable, args: List, kwargs: Dict) -> Dict[str, Any]:
        """
        Validate function call against its type hints
        """
        try:
            signature = inspect.signature(func)
            type_hints = get_type_hints(func)
            
            # Bind arguments
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            violations = []
            
            # Check each argument
            for param_name, param_value in bound_args.arguments.items():
                if param_name in type_hints:
                    expected_type = type_hints[param_name]
                    
                    # Skip Any type
                    if expected_type == Any:
                        continue
                    
                    # Check if value matches expected type
                    if not isinstance(param_value, expected_type):
                        violations.append(
                            f"Parameter '{param_name}': expected {expected_type}, "
                            f"got {type(param_value)}"
                        )
            
            return {
                'is_valid': len(violations) == 0,
                'violations': violations,
                'parameters_checked': len(bound_args.arguments),
                'has_type_hints': len(type_hints) > 0
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'violations': [f"Signature validation error: {str(e)}"],
                'error': str(e)
            }
    
    @staticmethod
    def create_schema_from_dataframe(df: pd.DataFrame) -> Dict[str, str]:
        """
        Create a schema definition from an existing DataFrame
        """
        schema = {}
        for col_name, dtype in df.dtypes.items():
            schema[col_name] = str(dtype)
        return schema
