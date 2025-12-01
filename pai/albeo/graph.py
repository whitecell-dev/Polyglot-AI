"""
Graph Algebra - Neighbor traversal, subgraph filtering
"""
import pandas as pd
import networkx as nx
from typing import List, Dict, Any


def graph_neighbors(df: pd.DataFrame, src: str, dst: str) -> pd.DataFrame:
    """
    Build a directed edge graph and return a DataFrame of:
       node_id | neighbors (list)

    df: edge list DataFrame
    src: source column name
    dst: destination column name
    """
    df = df.copy()

    # Build graph from edge list
    G = nx.from_pandas_edgelist(df, src, dst, create_using=nx.DiGraph())

    # Get unique nodes from both columns
    all_nodes = pd.Index(
        pd.concat([df[src], df[dst]]).unique(),
        name='node_id'
    )

    df_nodes = pd.DataFrame(index=all_nodes)

    # Mapping node → neighbor list
    neighbor_map = {n: list(G.neighbors(n)) for n in G.nodes()}

    df_nodes['neighbors'] = df_nodes.index.map(lambda x: neighbor_map.get(x, []))

    return df_nodes.reset_index()


def graph_depth_limited_paths(df: pd.DataFrame, src: str, dst: str,
                              root: Any, max_depth: int) -> List[List[Any]]:
    """
    Return depth-limited DFS paths starting from root.

    Output is strictly:
        List[List[node_ids]]
    """
    df = df.copy()
    G = nx.from_pandas_edgelist(df, src, dst, create_using=nx.DiGraph())

    paths = []

    def dfs(node, depth, path_so_far):
        if depth > max_depth:
            return
        new_path = path_so_far + [node]
        paths.append(new_path)
        for nbr in G.neighbors(node):
            dfs(nbr, depth + 1, new_path)

    dfs(root, 0, [])
    return paths
