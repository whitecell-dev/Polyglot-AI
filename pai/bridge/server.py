# pai/bridge/server.py
"""
pAI Bridge Server — FastAPI JSON-RPC Gateway
Lua / Go / Node → Python (ALBEO) via JSON-RPC

Runs the ALBEO layer as a computation microservice.
Only primitives cross the boundary.
"""

import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
import uvicorn
import pandas as pd

# Load bridge + ALBEO
from pai.bridge.pbridge import PandasStructuralBridge
from pai.bridge.validate import ensure_primitives
from pai.albeo import (
    constraint_solve,
    dataflow_split, dataflow_merge,
    graph_neighbors, graph_depth_limited_paths,
    json_lens_extract, json_normalize_column,
    calculate_ratios, calculate_aggregate_metrics, ml_feature_engineering,
    streaming_rolling, streaming_expanding, streaming_duckdb_window,
    temporal_lag_lead, temporal_resample, temporal_join
)

# Function registry — this is the API surface Lua can call
FUNCTIONS = {
    "shape_info": PandasStructuralBridge.shape_info,
    "schema_validation": PandasStructuralBridge.schema_validation,
    "basic_info": PandasStructuralBridge.basic_info,
    "get_column_values": PandasStructuralBridge.get_column_values,
    "json_lens_extract": PandasStructuralBridge.json_lens_extract,
    "extract_numerical_stats": PandasStructuralBridge.extract_numerical_stats,
    "describe_dataframe": PandasStructuralBridge.describe_dataframe,

    # ALBEO functions
    "calculate_ratios": calculate_ratios,
    "calculate_aggregate_metrics": calculate_aggregate_metrics,
    "ml_feature_engineering": ml_feature_engineering,
    "constraint_solve": constraint_solve,
    "temporal_lag_lead": temporal_lag_lead,
    "temporal_resample": temporal_resample,
    "temporal_join": temporal_join,
    "streaming_rolling": streaming_rolling,
    "streaming_expanding": streaming_expanding,
    "streaming_duckdb_window": streaming_duckdb_window,
    "graph_neighbors": graph_neighbors,
    "graph_depth_limited_paths": graph_depth_limited_paths,
}

# JSON-RPC request model
class RPCRequest(BaseModel):
    method: str
    params: Dict[str, Any]


app = FastAPI()


@app.post("/rpc")
def rpc_handler(req: RPCRequest):
    if req.method not in FUNCTIONS:
        return {"error": f"Unknown method '{req.method}'", "has_error": True}

    func = FUNCTIONS[req.method]

    # Extract DataFrame if provided
    params = req.params.copy()
    if "df" in params:
        params["df"] = pd.DataFrame(params["df"])  # Lua sends list-of-dicts

    try:
        result = func(**params)
        result = ensure_primitives(result)
        return {"result": result, "has_error": False}
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "has_error": True
        }


def run():
    uvicorn.run(app, host="127.0.0.1", port=7777)


if __name__ == "__main__":
    run()

