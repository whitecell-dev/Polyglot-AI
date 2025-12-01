import pandas as pd
from pai.bridge.pbridge import PandasStructuralBridge
from .albeo.scoring import calculate_ratios

df = pd.DataFrame({
    "income": [5000, 6000, 4000],
    "debt": [2000, 2500, 4500],
    "loan_amount": [200000, 180000, 220000],
    "credit_score": [680, 720, 580]
})

print("=== ALBEO Pure Computation ===")
computed = calculate_ratios(df)
print(computed)

print("\n=== Bridge: shape_info ===")
info = PandasStructuralBridge.shape_info(computed)
print(info)

print("\n=== Bridge: extract stats ===")
stats = PandasStructuralBridge.extract_numerical_stats(computed, "dti_ratio")
print(stats)

print("\n=== Bridge: convert to primitives ===")
from pai.bridge.validate import ensure_primitives
prims = ensure_primitives(computed)
print(prims)

