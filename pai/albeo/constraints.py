"""
Constraint Solving - Declarative rules solved via bounded search
"""
import pandas as pd
from typing import Dict


def constraint_solve(df: pd.DataFrame, constraints: Dict[str, str]) -> pd.DataFrame:
    """
    Evaluates named boolean constraints:
    constraints = {
        "adult": "age >= 18",
        "good_income": "income > debt * 0.4",
        "credit_ok": "credit_score >= 620"
    }

    Adds:
        meets_constraint_<name>
    plus:
        meets_all
    """
    df = df.copy()
    flags = []

    for name, expr in constraints.items():
        col_name = f"meets_constraint_{name}"
        df[col_name] = df.eval(expr)
        flags.append(df[col_name])

    df["meets_all"] = pd.concat(flags, axis=1).all(axis=1)
    return df
