"""
Validation utilities for bridge operations
"""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union, Type
import json


def ensure_primitives(obj: Any) -> Any:
    """
    Recursively convert object to primitives only
    Used before crossing language boundaries
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        # Convert DataFrame to list of dicts (primitives only)
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {k: ensure_primitives(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_primitives(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(ensure_primitives(item) for item in obj)
    elif hasattr(obj, 'to_dict'):
        # Handle objects with to_dict method (like dataclasses)
        return ensure_primitives(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        # Handle regular objects
        return ensure_primitives(obj.__dict__)
    else:
        # Last resort: string representation
        return str(obj)


def validate_bridge_call(func_name: str, args: List, kwargs: Dict, 
                        allowed_functions: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Validate that a bridge call is allowed and properly formed
    """
    if func_name not in allowed_functions:
        return {
            'is_valid': False,
            'error': f"Function '{func_name}' is not allowed via bridge",
            'allowed_functions': list(allowed_functions.keys())
        }
    
    func_spec = allowed_functions[func_name]
    
    # Check argument count
    expected_args = func_spec.get('args', [])
    min_args = func_spec.get('min_args', len(expected_args))
    max_args = func_spec.get('max_args', len(expected_args))
    
    total_args = len(args) + len(kwargs)
    
    if total_args < min_args or total_args > max_args:
        return {
            'is_valid': False,
            'error': f"Function '{func_name}' expects {min_args}-{max_args} args, got {total_args}",
            'expected_args': expected_args
        }
    
    # Check that all arguments are primitives
    all_args = list(args) + list(kwargs.values())
    non_primitives = []
    
    for i, arg in enumerate(all_args):
        if not is_primitive(arg):
            non_primitives.append({
                'position': i,
                'type': type(arg).__name__,
                'value': str(arg)[:100]  # Truncate for safety
            })
    
    if non_primitives:
        return {
            'is_valid': False,
            'error': f"Function '{func_name}' received non-primitive arguments",
            'non_primitives': non_primitives
        }
    
    return {
        'is_valid': True,
        'func_name': func_name,
        'arg_count': total_args,
        'has_kwargs': len(kwargs) > 0
    }


def is_primitive(obj: Any) -> bool:
    """
    Check if an object is a primitive type
    """
    if obj is None:
        return True
    elif isinstance(obj, (str, int, float, bool)):
        return True
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return True
    elif isinstance(obj, (list, tuple, dict)):
        # Recursively check contents
        if isinstance(obj, (list, tuple)):
            return all(is_primitive(item) for item in obj)
        elif isinstance(obj, dict):
            return all(is_primitive(k) and is_primitive(v) for k, v in obj.items())
    return False


def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize object to JSON, converting non-serializable parts
    """
    def default_converter(o):
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return float(o) if isinstance(o, np.floating) else int(o)
        elif isinstance(o, pd.Timestamp):
            return o.isoformat()
        elif hasattr(o, 'to_dict'):
            return o.to_dict()
        elif hasattr(o, '__dict__'):
            return o.__dict__
        else:
            return str(o)
    
    return json.dumps(obj, default=default_converter)
