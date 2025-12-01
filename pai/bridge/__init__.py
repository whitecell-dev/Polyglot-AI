"""
Bridge Layer – typed boundary between IMPO runtimes and ALBEO computation.
Never expose Pandas or non-primitive Python objects across the boundary.
"""

from .pbridge import PandasStructuralBridge
from .error_bridge import pAIError, pAIErrorBridge
from .schema_bridge import SchemaValidator
from .validate import validate_bridge_call, ensure_primitives

__all__ = [
    "PandasStructuralBridge",
    "pAIError",
    "pAIErrorBridge",
    "SchemaValidator",
    "validate_bridge_call",
    "ensure_primitives"
]

