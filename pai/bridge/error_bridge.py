"""
Error Bridge - Unified error handling across language boundaries
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional
import traceback


@dataclass
class pAIError:
    """Unified error representation across Python/IMPO boundary"""
    source: str  # 'albeo', 'impo', 'bridge'
    component: str  # Function/module name
    error_type: str  # Exception type
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    
    def __str__(self):
        return f"[pAI/{self.source}/{self.component}] {self.error_type}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to primitive dict for cross-language transmission"""
        return {
            'source': self.source,
            'component': self.component,
            'error_type': self.error_type,
            'message': self.message,
            'context': self.context,
            'stack_trace': self.stack_trace,
            'has_error': True
        }


class pAIErrorBridge:
    """
    ERROR BOUNDARY: Translates errors between Python and IMPO layers
    Ensures errors don't leak complex objects across the boundary
    """
    
    @staticmethod
    def python_to_impo_error(error: Exception, component: str, 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Convert Python exception to IMPO-safe dict
        Returns: Simple dict with primitives only
        """
        return {
            'source': 'albeo',
            'component': component,
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context or {},
            'stack_trace': traceback.format_exc(),
            'has_error': True
        }
    
    @staticmethod
    def impo_to_python_error(impo_error_dict: Dict) -> pAIError:
        """
        Convert IMPO error dict to Python pAIError
        """
        return pAIError(
            source=impo_error_dict.get('source', 'impo'),
            component=impo_error_dict.get('component', 'unknown'),
            error_type=impo_error_dict.get('error_type', 'IMPOError'),
            message=impo_error_dict.get('message', 'Unknown error'),
            context=impo_error_dict.get('context', {}),
            stack_trace=impo_error_dict.get('stack_trace')
        )
    
    @staticmethod
    def create_error(source: str, component: str, error_type: str, 
                    message: str, context: Dict[str, Any] = None) -> pAIError:
        """
        Create a structured pAIError
        """
        return pAIError(
            source=source,
            component=component,
            error_type=error_type,
            message=message,
            context=context or {},
            stack_trace=traceback.format_stack()
        )
