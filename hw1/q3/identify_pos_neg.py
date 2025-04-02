#!/usr/bin/env python3
import sys, os, math, subprocess, tempfile, shutil, random
import networkx as nx
from math import log
from concurrent.futures import ProcessPoolExecutor

EPS = 1e-4

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
            if not line: 
                continue
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

def add_reverse_edges_to_lines(lines):
    out_lines = []
    for line in lines:
        out_lines.append(line)
        if line.startswith("e"):
            parts = line.split()
            u = parts[1]
            v = parts[2]
            label = parts[3]
            if u != v:
                out_lines.append(f"e {v} {u} {label}")
    return out_lines

def node_match(n1, n2):
    return n1.get('label') == n2.get('label')

def edge_match(e1, e2):
    return e1.get('label') == e2.get('label')

# def compute_support(pattern_nx, graphs_nx):
#     count = 0
#     # add some prechecks
    
#     for G in graphs_nx:
#         matcher = nx.algorithms.isomorphism.GraphMatcher(G, pattern_nx,
#                                                          node_match=node_match,
#                                                          edge_match=edge_match)
#         if matcher.subgraph_is_isomorphic():
#             count += 1
#     return count

# def process_and_select_patterns(gaston_output_file, pos_graphs, neg_graphs, top_n=100):
#     raw_patterns = parse_gaston_output(gaston_output_file)
#     if not raw_patterns:
#         print("No patterns found.")
#         return []
#     candidates = []
#     for support_dummy, lines in raw_patterns:
#         proc_lines = add_reverse_edges_to_lines(lines)
#         pattern_nx = parse_pattern_lines_to_graph(proc_lines)
#         neg_sup = compute_support(pattern_nx, neg_graphs)
#         disc_score = log((support_dummy/len(pos_graphs) + EPS) / (neg_sup/len(neg_graphs) + EPS))
#         print(f"Pattern processed: pos_sup={support_dummy}, neg_sup={neg_sup}, score={disc_score:.4f}")
#         candidates.append((disc_score, proc_lines))
        
        
#     candidates_sorted = sorted(candidates, key=lambda x: x[0], reverse=True)
#     selected = candidates_sorted[:top_n]
#     processed_patterns = []
#     for score, lines in selected:
#         out_lines = [f"# {score:.4f}"]
#         for line in lines:
#             out_lines.append(line)
#         processed_patterns.append(out_lines)
#     return processed_patterns


def compute_support_seq(pattern_nx, graphs_nx):
    count = 0
    for G in graphs_nx:
        matcher = nx.algorithms.isomorphism.GraphMatcher(G, pattern_nx,
                                                         node_match=node_match,
                                                         edge_match=edge_match)
        if matcher.subgraph_is_isomorphic():
            count += 1
    return count

def process_candidate(candidate, pos_graphs, neg_graphs):
    support_dummy, lines = candidate
    proc_lines = add_reverse_edges_to_lines(lines)
    pattern_nx = parse_pattern_lines_to_graph(proc_lines)
    pos_sup = support_dummy  # Use Gaston output as positive support
    neg_sup = compute_support_seq(pattern_nx, neg_graphs)
    disc_score = log(pow((pos_sup/len(pos_graphs) + EPS),2) / (neg_sup/len(neg_graphs) + EPS))
    print(f"Candidate processed: pos_sup={pos_sup}, neg_sup={neg_sup}, score={disc_score:.4f}")
    return (disc_score, proc_lines)

def process_candidate_chunk(chunk, pos_graphs, neg_graphs):
    results = []
    for candidate in chunk:
        results.append(process_candidate(candidate, pos_graphs, neg_graphs))
    return results

def process_and_select_patterns(gaston_output_file, pos_graphs, neg_graphs, top_n=100, num_chunks=4):
    raw_patterns = parse_gaston_output(gaston_output_file)
    if not raw_patterns:
        print("No patterns found.")
        return []
    # Split raw_patterns into num_chunks parts
    chunk_size = math.ceil(len(raw_patterns) / num_chunks)
    chunks = [raw_patterns[i:i+chunk_size] for i in range(0, len(raw_patterns), chunk_size)]
    candidates = []
    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        futures = [executor.submit(process_candidate_chunk, chunk, pos_graphs, neg_graphs) for chunk in chunks]
        for fut in futures:
            candidates.extend(fut.result())
    candidates_sorted = sorted(candidates, key=lambda x: x[0], reverse=True)
    
    selected = candidates_sorted[:top_n]
    processed_patterns = []
    for score, lines in selected:
        out_lines = [f"# {score:.4f}"]
        for line in lines:
            out_lines.append(line)
        processed_patterns.append(out_lines)
    return processed_patterns


def write_processed_patterns(processed_patterns, output_file):
    with open(output_file, "w") as f:
        for pattern in processed_patterns:
            for line in pattern:
                f.write(line + "\n")
    print(f"Patterns written to {output_file}")

def main():
    if len(sys.argv) != 6:
        sys.exit("Usage: python mine.py <graph_file> <labels_file> <output_pattern_file> <m> <support_ratio>")
    graph_file = sys.argv[1]
    labels_file = sys.argv[2]
    final_output_file = sys.argv[3]
    m = int(sys.argv[4])
    sup_ratio = float(sys.argv[5])
    graphs = read_graphs(graph_file)
    labels = read_labels(labels_file)
    if len(graphs) != len(labels):
        sys.exit("Error: Number of graphs and labels do not match.")
    positive_graphs = [g for g, lab in zip(graphs, labels) if lab == "1"]
    negative_graphs = [g for g, lab in zip(graphs, labels) if lab == "0"]
    print(f"Total graphs: {len(graphs)}")
    print(f"Positive graphs (label 1): {len(positive_graphs)}")
    print(f"Negative graphs (label 0): {len(negative_graphs)}")
    min_sup_pos = math.ceil(sup_ratio * len(positive_graphs))
    print(f"Positive min support: {min_sup_pos}, m: {m}")
    tmp_dir = tempfile.mkdtemp()
    pos_input_file = os.path.join(tmp_dir, "positive_graphs.txt")
    gaston_output_file = os.path.join(tmp_dir, "patterns.txt")
    write_graphs_to_file(positive_graphs, pos_input_file)
    print("Running Gaston on positive graphs...")
    run_gaston(pos_input_file, min_sup_pos, gaston_output_file, m)
    processed_patterns = process_and_select_patterns(gaston_output_file, positive_graphs, negative_graphs, top_n=100)
    write_processed_patterns(processed_patterns, final_output_file)
    shutil.rmtree(tmp_dir)
    print(f"Patterns written to {final_output_file}")

if __name__ == "__main__":
    main()

