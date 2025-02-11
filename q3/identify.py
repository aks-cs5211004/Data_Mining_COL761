import math
import os
import shutil
import subprocess
import sys
import tempfile

import networkx as nx


def read_graphs(graph_file):
    graphs = []
    current_graph = None
    with open(graph_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if current_graph is not None:
                    graphs.append(current_graph)
                current_graph = nx.Graph()
            elif line.startswith("v"):
                if current_graph is None:
                    current_graph = nx.Graph()
                parts = line.split()
                try:
                    node_id = int(parts[1])
                    label = parts[2]
                    current_graph.add_node(node_id, label=label)
                except Exception as e:
                    print(f"Error processing node line '{line}': {e}", file=sys.stderr)
            elif line.startswith("e"):
                if current_graph is None:
                    current_graph = nx.Graph()
                parts = line.split()
                try:
                    u = int(parts[1])
                    v = int(parts[2])
                    label = parts[3]
                    current_graph.add_edge(u, v, label=label)
                except Exception as e:
                    print(f"Error processing edge line '{line}': {e}", file=sys.stderr)
        if current_graph is not None:
            graphs.append(current_graph)
    return graphs

def read_labels(labels_file):
    with open(labels_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def write_graphs_to_file(graphs, filename):
    with open(filename, 'w') as f:
        for idx, G in enumerate(graphs):
            f.write(f"t # {idx}\n")
            for n in sorted(G.nodes()):
                f.write(f"v {n} {G.nodes[n]['label']}\n")
            for u, v, data in G.edges(data=True):
                f.write(f"e {u} {v} {data['label']}\n")

def run_gaston(input_file, min_sup, output_file, m):
    command = ["./Binaries/gaston", "-m", str(m), str(min_sup), input_file, output_file]
    subprocess.run(command, check=True)
    if not os.path.exists(output_file):
        sys.exit(f"Error: Output file {output_file} not found.")

def parse_gaston_output(input_file):
    patterns = []
    with open(input_file, "r") as f:
        current_support = None
        current_lines = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("#"):
                if current_support is not None and current_lines:
                    try:
                        support = int(current_support)
                    except:
                        support = 0
                    patterns.append((support, current_lines))
                current_support = line[1:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_support is not None and current_lines:
            try:
                support = int(current_support)
            except:
                support = 0
            patterns.append((support, current_lines))
    return patterns

def parse_pattern_lines_to_graph(lines):
    g = nx.Graph()
    for line in lines:
        if line.startswith("v"):
            parts = line.split()
            node_id = int(parts[1])
            label = parts[2]
            g.add_node(node_id, label=label)
        elif line.startswith("e"):
            parts = line.split()
            u = int(parts[1])
            v = int(parts[2])
            label = parts[3]
            g.add_edge(u, v, label=label)
    return g

def process_and_select_patterns(gaston_output_file, top_n=100):
    patterns = parse_gaston_output(gaston_output_file)
    if not patterns:
        print("No patterns found.")
        return []
    patterns_sorted = sorted(patterns, key=lambda x: x[0], reverse=True)
    selected = patterns_sorted[:top_n]
    processed_patterns = []
    for support, lines in selected:
        g = parse_pattern_lines_to_graph(lines)
        out_lines = [f"# {support}"]
        for n in sorted(g.nodes()):
            out_lines.append(f"v {n} {g.nodes[n]['label']}")
        for u, v, data in sorted(g.edges(data=True)):
            out_lines.append(f"e {u} {v} {data['label']}")
            if u != v:
                out_lines.append(f"e {v} {u} {data['label']}")
        processed_patterns.append(out_lines)
    return processed_patterns

def write_processed_patterns(processed_patterns, output_file):
    with open(output_file, "w") as f:
        for pattern in processed_patterns:
            for line in pattern:
                f.write(line + "\n")
    print(f"Patterns written to {output_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_file", help="Path to graphs.txt")
    parser.add_argument("labels_file", help="Path to labels.txt") 
    parser.add_argument("output_pattern_file", help="Output pattern file")
    parser.add_argument("--m", type=int, default=5, help="Minimum pattern size (default 5)")
    parser.add_argument("--s", type=float, default=0.2, help="Support ratio (default 0.2)") #grid search 
    args = parser.parse_args()
    graphs = read_graphs(args.graph_file)
    labels = read_labels(args.labels_file)
    
    # Note to iterate over args.s and args.m and to choose postive as the lesser guy.
    # Mutageneicity 4 and 0.01
    # Others 5 and 0.1
    print(f"Read {len(graphs)} graphs and {len(labels)} labels.")
    if len(graphs) != len(labels):
        sys.exit("Error: Number of graphs and labels do not match.")
    positive_graphs = [g for g, lab in zip(graphs, labels) if lab == "1"]
    print(f"Total graphs: {len(graphs)}, Positive graphs (label 1): {len(positive_graphs)}")
    min_sup = math.ceil(args.s * len(positive_graphs))
    print(f"Minimum support: {min_sup}, m: {args.m}")
    tmp_dir = tempfile.mkdtemp()
    input_file = os.path.join(tmp_dir, "positive_graphs.txt")
    gaston_output_file = os.path.join(tmp_dir, "patterns.txt")
    write_graphs_to_file(positive_graphs, input_file)
    print("Running Gaston on positive graphs...")
    run_gaston(input_file, min_sup, gaston_output_file, args.m)
    processed_patterns = process_and_select_patterns(gaston_output_file, top_n=100)
    write_processed_patterns(processed_patterns, args.output_pattern_file)
    shutil.rmtree(tmp_dir)
    print(f"Patterns written to {args.output_pattern_file}")

if __name__ == "__main__":
    main()
