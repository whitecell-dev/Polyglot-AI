# Polyglot AI 

Deterministic polyglot runtime for AI-generated systems with enforced architectural boundaries.

## Overview

this provides a stable foundation for AI-generated code by enforcing clean separation between algebraic computation (ALBEO) and imperative orchestration (IMPO). The system prevents common AI-generated code failures through structural bridges that maintain runtime invariants across language boundaries.

## Architecture

```
ALBEO (Python) ↔ Structural Bridge ↔ IMPO (Lua)
     ↓                              ↓
Deterministic Computation    I/O & Orchestration
```

### Core Concepts

**ALBEO** - Algebraic business logic operations:
- Pure, deterministic computations
- Vectorized data transformations (Pandas/NumPy)
- Schema-native operations
- No side effects, no I/O

**IMPO** - Imperative orchestration layer:
- External API calls with retry logic
- File I/O sequences
- Workflow state management
- Temporal operations and scheduling

**Structural Bridge** - Enforced boundary layer:
- Converts complex objects to primitives
- Prevents cross-language leakage
- Maintains runtime invariants
- Provides unified error handling

## Quick Start

```bash
# Clone repository
git clone https://github.com/whitecell-dev/polyglot-ai
cd Polyglot-AI

# Install dependencies
pip install -r requirements.txt

# Run reference implementation
python core/bridge_reference.py
```

## Core Components

```
polyglot-ai/
├── core/
│   ├── bridge_reference.py          # Canonical implementation
│   ├── base_engine.py               # Base runtime classes
│   └── types.py                     # Shared type definitions
├── templates/
│   ├── impo_functions.lua           # IMPO function templates
│   └── albeo_functions.py           # ALBEO pattern templates
└── examples/
    └── production_workflow.py       # Complete integration example
```

## Usage Example

```python
from pAI.core import pAIimperativeEngine, pAIAlgebraicCore

# Initialize runtime
engine = pAIImperativeEngine()
albeo = pAIAlgebraicCore()

# ALBEO: Pure computation
applications_with_ratios = albeo.calculate_financial_ratios(applications)

# IMPO: External orchestration  
validation = engine.validate_dataframe_structure(applications, expected_columns)
api_result = engine.external_api_call("CreditBureau_123", 3, 100)

# Bridge maintains boundary integrity
assert validation['structural_valid'] == True
assert isinstance(api_result['success'], bool)  # Primitives only
```

## Bridge Enforcement

The structural bridge enforces four critical functions:

1. **Configuration Purity** - Minimal, stable API surface
2. **Data Semantics Mapping** - Type-safe primitive conversion
3. **Syntax Enforcement** - LLM-safe patterns only
4. **Error Boundary** - Unified cross-language error handling

## AI Generation Safety

prevents common AI-generated code failures:

- **No object leakage** - Complex objects never cross language boundaries
- **Immutable data flow** - ALBEO operations never mutate input data
- **Deterministic execution** - IMPO handles all non-deterministic operations
- **Stable interfaces** - Bridge functions provide consistent primitives

## Production Patterns

```python
# ALBEO: Always immutable
def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Critical: never mutate input
    df['dti_ratio'] = (df['debt'] / df['income']) * 100
    return df

# IMPO: Only orchestration
def external_api_call(api_name: str, max_retries: int) -> Dict[str, Any]:
    # Returns only primitives: bool, int, str, list, dict
    return {'success': True, 'attempts': 1}
```

## Requirements

- Python 3.8+
- Lua 5.3+ (via lupa)
- Pandas 1.3+
- NumPy 1.20+

## Dependencies

- lupa: Python-Lua bridge
- pandas: Data manipulation
- numpy: Numerical operations

## License

MIT
