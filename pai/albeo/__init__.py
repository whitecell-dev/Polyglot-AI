"""
ALBEO – Pure Algebraic Computation Layer
Pure Pandas/Numpy algebra with no side effects.
"""

from .constraints import constraint_solve
from .dataflow import (
    dataflow_split,
    dataflow_merge,
    dataflow_tee,
    dataflow_failover
)
from .graph import (
    graph_neighbors,
    graph_depth_limited_paths
)
from .lenses import (
    json_lens_extract,
    json_normalize_column
)
from .scoring import (
    calculate_ratios,
    calculate_aggregate_metrics,
    ml_feature_engineering
)
from .streaming import (
    streaming_rolling,
    streaming_expanding,
    streaming_duckdb_window
)
from .temporal import (
    temporal_lag_lead,
    temporal_resample,
    temporal_join
)

__all__ = [
    "constraint_solve",
    "dataflow_split", "dataflow_merge", "dataflow_tee", "dataflow_failover",
    "graph_neighbors", "graph_depth_limited_paths",
    "json_lens_extract", "json_normalize_column",
    "calculate_ratios", "calculate_aggregate_metrics", "ml_feature_engineering",
    "streaming_rolling", "streaming_expanding", "streaming_duckdb_window",
    "temporal_lag_lead", "temporal_resample", "temporal_join"
]

