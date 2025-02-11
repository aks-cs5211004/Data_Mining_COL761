import argparse
import os

import numpy as np
import rustworkx as rx
from joblib import Parallel, delayed


def load_graphs(file_path):
    """Load graphs with proper attribute handling and validation"""
    graphs = []
    with open(file_path, 'r') as f:
        graph_data = f.read().strip().split("#")
    
    for idx, graph_str in enumerate(graph_data):
        if not graph_str.strip():
            continue
            
        graph = rx.PyDiGraph(multigraph=False)  # Enable parallel edges
        node_map = {}
        lines = [ln.strip() for ln in graph_str.split('\n') if ln.strip()]
        
        for line in lines:
            parts = line.split()
            if parts[0] == 'v':
                try:
                    node_id = int(parts[1])
                    label = int(parts[2])
                    node_idx = graph.add_node({'label': label})
                    node_map[node_id] = node_idx
                except (IndexError, ValueError) as e:
                    raise ValueError(f"Invalid node format in graph {idx}: {line}") from e
                    
            elif parts[0] == 'e':
                try:
                    u = int(parts[1])
                    v = int(parts[2])
                    label = int(parts[3])
                    graph.add_edge(node_map[u], node_map[v], {'label': label})
                except KeyError as e:
                    raise ValueError(f"Edge references undefined node in graph {idx}: {line}") from e
                except IndexError as e:
                    raise ValueError(f"Invalid edge format in graph {idx}: {line}") from e
        
        if graph.num_nodes() == 0:
            raise ValueError(f"Empty graph detected at position {idx}")
        graphs.append(graph)
    
    return graphs

def check_subgraph(graph, subgraph):
    """Enhanced subgraph isomorphism with validation"""
    return rx.is_subgraph_isomorphic(
        graph, subgraph,
        node_matcher=lambda a, b: a['label'] == b['label'],
        edge_matcher=lambda a, b: a['label'] == b['label']
        # induced=False,
        # id_order=False
    )

def process_graph(i, graph, subgraphs):
    """Feature extraction with debug logging"""
    features = np.zeros(len(subgraphs), dtype=int)
    for j, sg in enumerate(subgraphs):
        if check_subgraph(graph, sg):
            features[j] = 1
            # print(f"Match found: Graph {i} contains subgraph {j}")  # Debug output
    return i, features

def convert_graphs_to_features(graphs, subgraphs, n_jobs):
    """Parallel processing with order preservation"""
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_graph)(i, g, subgraphs)
        for i, g in enumerate(graphs)
    ) 
    return np.array([f for _, f in sorted(results)])

def save_features(features, output_base):
    """Save features with proper CSV formatting"""
    np.save(f"{output_base}.npy", features)
    np.savetxt(f"{output_base}.csv", features, 
               fmt='%d', delimiter=',', header='', comments='')

def validate_inputs(graphs, subgraphs):
    """Comprehensive input validation"""
    if not graphs:
        raise ValueError("No graphs loaded from input file")
    if not subgraphs:
        raise ValueError("No subgraphs loaded from pattern file")
    
    sample_graph = graphs[0]
    sample_subgraph = subgraphs[0]
    
    # Verify label types
    assert isinstance(sample_graph.get_node_data(0)['label'], int), "Node labels must be integers"
    assert isinstance(sample_subgraph.get_edge_data(0,1)['label'], int), "Edge labels must be integers"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Feature Extractor")
    parser.add_argument("graph_file", help="Input graphs file path")
    parser.add_argument("subgraph_file", help="Subgraph patterns file path")
    parser.add_argument("output_base", help="Base name for output files")
    parser.add_argument("--n_jobs", type=int, default=-1, 
                       help="Number of parallel jobs (default: all cores)")
    args = parser.parse_args()

    try:
        print("Loading graphs...")
        graphs = load_graphs(args.graph_file)
        print(f"Loaded {len(graphs)} graphs")
        print("Loading subgraphs...")
        subgraphs = load_graphs(args.subgraph_file)
        
        print("Validating inputs...")
        validate_inputs(graphs, subgraphs)
        
        print("Generating features...")
        features = convert_graphs_to_features(graphs, subgraphs, args.n_jobs)
        
        if np.all(features == 0):
            print("\nERROR: No subgraph matches found. Verify:")
            print("- Subgraphs exist in input graphs")
            print("- Node/edge labels match exactly")
            print("- Subgraphs are smaller than input graphs")
            print("- Input files follow the format: v <id> <label> / e <id1> <id2> <label>")
        else:
            print(f"\nSuccess! Found {np.sum(features)} total matches")
            save_features(features, args.output_base)
            print(f"Saved features to {args.output_base}.npy and {args.output_base}.csv")
            
    except Exception as e:
        print(f"\nFatal Error: {str(e)}")
        print("Check input files and run with --help for usage information")
