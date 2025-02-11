#!/usr/bin/env python3
import sys, os, math, subprocess, tempfile, heapq, pickle
from copy import deepcopy
import networkx as nx
from networkx.algorithms import isomorphism
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
import multiprocessing
from functools import partial
from collections import deque

EPS = 1e-6
MIN_SCORE_THRESHOLD = 1.0
MIN_SUPPORT_PERCENT = 0.1
MAX_PATTERNS = 100

def check_graph_for_candidate(candidate_graph, G):
    nm = isomorphism.categorical_node_match('label', None)
    em = isomorphism.categorical_edge_match('label', None)
    matcher = isomorphism.GraphMatcher(G, candidate_graph, node_match=nm, edge_match=em)
    return matcher.subgraph_is_isomorphic()

def write_graphs_to_file(graphs, filename):
    with open(filename, 'w') as f:
        for idx, g in enumerate(graphs):
            f.write("t # {}\n".format(idx))
            for n in sorted(g.nodes()):
                f.write("v {} {}\n".format(n, g.nodes[n]['label']))
            for u, v, data in g.edges(data=True):
                f.write("e {} {} {}\n".format(u, v, data['label']))

def run_gaston(input_file, min_sup, output_file):
    command = ["./gaston", str(min_sup), input_file, output_file]
    subprocess.run(command, check=True)
    if not os.path.exists(output_file):
        sys.exit("Error: Gaston did not produce an output file: " + output_file)

def parse_pattern_lines_to_graph(lines):
    g = nx.Graph()
    for line in lines:
        if line.startswith("v"):
            parts = line.split()
            node_id = int(parts[1])
            g.add_node(node_id, label=parts[2])
        elif line.startswith("e"):
            parts = line.split()
            u = int(parts[1])
            v = int(parts[2])
            g.add_edge(u, v, label=parts[3])
    return g

def parse_gaston_output(filename):
    supports = {}
    with open(filename, "r") as f:
        current_support = None
        current_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if current_support is not None and current_lines:
                    g = parse_pattern_lines_to_graph(current_lines)
                    cc = weisfeiler_lehman_graph_hash(g, edge_attr='label', node_attr='label')
                    supports[cc] = int(current_support)
                current_support = line[1:].strip()
                current_lines = []
            elif line.startswith("t"):
                continue
            else:
                current_lines.append(line)
        if current_support is not None and current_lines:
            g = parse_pattern_lines_to_graph(current_lines)
            cc = weisfeiler_lehman_graph_hash(g, edge_attr='label', node_attr='label')
            supports[cc] = int(current_support)
    return supports

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
                parts = line.split()
                node_id = int(parts[1])
                label = parts[2]
                current_graph.add_node(node_id, label=label)
            elif line.startswith("e"):
                parts = line.split()
                u = int(parts[1])
                v = int(parts[2])
                label = parts[3]
                current_graph.add_edge(u, v, label=label)
        if current_graph is not None:
            graphs.append(current_graph)
    return graphs

def read_labels(labels_file):
    labels = []
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                labels.append(int(line))
    return labels

class Environment:
    def __init__(self):
        self.possible_edges = {}
        self.connecting_nodes = {}
        self.generated_patterns = set()
        self.pos_supports = {}
        self.neg_supports = {}
        self.total_positive = 0
        self.total_negative = 0

    def build_domain_info(self, positive_graphs):
        self.possible_edges.clear()
        self.connecting_nodes.clear()
        for G in positive_graphs:
            for u, v, data in G.edges(data=True):
                label_u = G.nodes[u].get('label')
                label_v = G.nodes[v].get('label')
                edge_label = data.get('label')
                if label_u is None or label_v is None or edge_label is None:
                    continue
                for key in [(label_u, label_v), (label_v, label_u)]:
                    self.possible_edges.setdefault(key, set()).add(edge_label)
                self.connecting_nodes.setdefault(label_u, set()).add(label_v)
                self.connecting_nodes.setdefault(label_v, set()).add(label_u)
        print("Domain info computed:")
        for key, val in self.possible_edges.items():
            print(f"{key} -> {sorted(val)}")
        for label, conns in self.connecting_nodes.items():
            print(f"{label} -> {sorted(conns)}")
    
    def pattern_already_generated(self, ccam_code):
        return ccam_code in self.generated_patterns
    
    def add_generated_pattern(self, ccam_code):
        self.generated_patterns.add(ccam_code)
    
    def get_possible_edges(self, label1, label2):
        return list(self.possible_edges.get((label1, label2), []))
    
    def get_possible_connecting_nodes(self, label):
        return self.connecting_nodes.get(label, set())
    
    def load_dataset(self, graphs_file, labels_file):
        all_graphs = []
        current_graph = None
        with open(graphs_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if current_graph is not None:
                        all_graphs.append(current_graph)
                    current_graph = nx.Graph()
                elif line.startswith("v"):
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    node_id = int(parts[1])
                    node_label = parts[2]
                    current_graph.add_node(node_id, label=node_label)
                elif line.startswith("e"):
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    u = int(parts[1])
                    v = int(parts[2])
                    edge_label = parts[3]
                    current_graph.add_edge(u, v, label=edge_label)
            if current_graph is not None:
                all_graphs.append(current_graph)
        with open(labels_file, "r") as f:
            labels = [int(line.strip()) for line in f if line.strip() != ""]
        if len(all_graphs) != len(labels):
            sys.exit("Error: Number of graphs and labels do not match!")
        positive_graphs = [g for g, lab in zip(all_graphs, labels) if lab == 0]
        negative_graphs = [g for g, lab in zip(all_graphs, labels) if lab != 0]
        self.total_positive = len(positive_graphs)
        self.total_negative = len(negative_graphs)
        print(f"Dataset loaded: {len(positive_graphs)} positive graphs, {len(negative_graphs)} negative graphs.")
        return positive_graphs, negative_graphs
    
    def load_gaston_supports(self, pos_support_file, neg_support_file):
        self.pos_supports = parse_gaston_output(pos_support_file)
        self.neg_supports = parse_gaston_output(neg_support_file)
        print(f"Gaston supports loaded: {len(self.pos_supports)} positive patterns, {len(self.neg_supports)} negative patterns.")
    
    def write_subgraphs(self, patterns, filename):
        with open(filename, "w") as f:
            for idx, pat in enumerate(patterns):
                f.write(f"t # {idx}\n")
                for n in sorted(pat.graph.nodes()):
                    f.write(f"v {n} {pat.graph.nodes[n]['label']}\n")
                for u, v, data in sorted(pat.graph.edges(data=True)):
                    if u > v:
                        u, v = v, u
                    f.write(f"e {u} {v} {data['label']}\n")
        print(f"Subgraphs written to {filename}")

class Pattern:
    def __init__(self, env, parent=None, graph=None):
        self.environment = env
        self.graph = nx.Graph() if graph is None else graph.copy()
        self.parent = parent
        self.discrimination_score = -math.inf
        self.pos_support = -1
        self.neg_support = -1
        self.loose_upper_bound = -math.inf
        self.ccam_code = ""
        self.edge_count = len(self.graph.edges)
    
    def calc_discrimination_score(self):
        if self.discrimination_score == -math.inf:
            self.pos_support = self.compute_support_pos()
            self.neg_support = self.compute_support_neg()
            self.discrimination_score = math.log(self.pos_support + EPS) - math.log(self.neg_support + EPS)
        return self.discrimination_score
    
    def compute_support_pos(self):
        cc = self.get_ccam_code()
        support = self.environment.pos_supports.get(cc, 0)
        self.pos_support = support / self.environment.total_positive if self.environment.total_positive > 0 else 0
        return self.pos_support
    
    def compute_support_neg(self):
        cc = self.get_ccam_code()
        support = self.environment.neg_supports.get(cc, 0)
        self.neg_support = support / self.environment.total_negative if self.environment.total_negative > 0 else 0
        return self.neg_support
    
    def get_loose_upper_bound(self):
        if self.loose_upper_bound == -math.inf:
            self.loose_upper_bound = math.log(self.pos_support + EPS) - math.log(EPS)
        return self.loose_upper_bound
    
    def get_edge_count(self):
        return self.edge_count

    def __lt__(self, other):
        return self.edge_count < other.edge_count
    
    def get_ccam_code(self):
        if not self.ccam_code:
            self.ccam_code = weisfeiler_lehman_graph_hash(self.graph, edge_attr='label', node_attr='label')
        return self.ccam_code
    
    def __str__(self):
        return f"Pattern(CCAM: {self.get_ccam_code()}, Score: {self.calc_discrimination_score():.4f})"
    
    def generate_child_patterns(self):
        children = []
        nodes_list = list(self.graph.nodes())
        for i in range(1, len(nodes_list)):
            for j in range(i):
                if not self.graph.has_edge(nodes_list[i], nodes_list[j]):
                    label_i = self.graph.nodes[nodes_list[i]]['label']
                    label_j = self.graph.nodes[nodes_list[j]]['label']
                    for edge_label in self.environment.get_possible_edges(label_i, label_j):
                        child = Pattern(self.environment, parent=self, graph=self.graph)
                        child.graph.add_edge(nodes_list[i], nodes_list[j], label=edge_label)
                        cc = child.get_ccam_code()
                        if not self.environment.pattern_already_generated(cc):
                            self.environment.add_generated_pattern(cc)
                            children.append(child)
        for node in nodes_list:
            label = self.graph.nodes[node]['label']
            for new_label in self.environment.get_possible_connecting_nodes(label):
                for edge_label in self.environment.get_possible_edges(label, new_label):
                    child = Pattern(self.environment, parent=self, graph=self.graph)
                    new_node_id = max(child.graph.nodes()) + 1 if child.graph.nodes() else 0
                    child.graph.add_node(new_node_id, label=new_label)
                    child.graph.add_edge(node, new_node_id, label=edge_label)
                    cc = child.get_ccam_code()
                    if not self.environment.pattern_already_generated(cc):
                        self.environment.add_generated_pattern(cc)
                        children.append(child)
        return children

POSITIVE_GRAPHS = []
NEGATIVE_GRAPHS = []



def fast_probe(graphs, labels, env):
    global POSITIVE_GRAPHS, NEGATIVE_GRAPHS
    POSITIVE_GRAPHS = [G for G, lab in zip(graphs, labels) if lab == 0]
    NEGATIVE_GRAPHS = [G for G, lab in zip(graphs, labels) if lab == 1]
    env.generated_patterns = set()
    candidates = []
    optimal_pattern = {}
    env.total_positive = len(POSITIVE_GRAPHS)
    env.total_negative = len(NEGATIVE_GRAPHS)
    for graph in POSITIVE_GRAPHS:
        for u, v, data in graph.edges(data=True):
            edge_label = data.get('label', '')
            base_graph = nx.Graph()
            base_graph.add_node(0, label=graph.nodes[u]['label'])
            base_graph.add_node(1, label=graph.nodes[v]['label'])
            base_graph.add_edge(0, 1, label=edge_label)
            candidate = Pattern(env, graph=base_graph)
            cc = candidate.get_ccam_code()
            if not env.pattern_already_generated(cc):
                env.add_generated_pattern(cc)
                candidates.append(candidate)
    print(f"Generated {len(candidates)} base candidates.")
    for G in POSITIVE_GRAPHS:
        optimal_pattern[G] = candidates[-1] if candidates else None
    candidate_queue = deque(candidates)
    processed = 0
    max_iter = 200
    print("Min optiimal: ", min(optimal_pattern.values(), key=lambda x: x.calc_discrimination_score()))
    while candidate_queue and processed < max_iter:
        current = candidate_queue.popleft()
        flag = False
        # print(f"Processing pattern: {current.get_ccam_code()} -> {current.calc_discrimination_score()}")
        for PG in POSITIVE_GRAPHS:
            if current.calc_discrimination_score() > optimal_pattern[PG].calc_discrimination_score() and check_graph_for_candidate(current.graph, PG):
                optimal_pattern[PG] = current
                flag = True
        if not flag:
            # processed += 1
            # print(f"Discard pattern: {current.get_ccam_code()} -> {current.calc_discrimination_score()}")
            continue
        # print("Min optiimal: ", min(optimal_pattern.values(), key=lambda x: x.calc_discrimination_score()))
        children = current.generate_child_patterns()
        candidate_queue.extend(children)
        processed += 1
        print(f"Processed {processed} candidates; queue size: {len(candidate_queue)}", end='\r')
    # sort in desc order of discrimination score. note that patterns should be unique first 
    unique_patterns = set()
    for pat in optimal_pattern.values():
        unique_patterns.add(pat.get_ccam_code())
    optimal_pattern = {G: pat for G, pat in optimal_pattern.items() if pat.get_ccam_code() in unique_patterns}
    print("\nOptimal patterns:")
    print("Min optiimal: ", min(optimal_pattern.values(), key=lambda x: x.calc_discrimination_score()))
    print("Max optiimal: ", max(optimal_pattern.values(), key=lambda x: x.calc_discrimination_score()))
    print(optimal_pattern)
    selected = sorted(optimal_pattern.values(), key=lambda x: x.calc_discrimination_score(), reverse=True)
    selected = selected[:MAX_PATTERNS]
    for idx, pat in enumerate(selected):
        print(f"Pattern {idx}: {pat.get_ccam_code()} -> {pat.calc_discrimination_score()}")
    print(f"\nSelected {len(selected)} discriminative patterns.")
    return selected

def main():
    if len(sys.argv) != 3:
        print("Usage: python lts.py <graph_file> <labels_file>")
        sys.exit(1)
    graph_file = sys.argv[1]
    labels_file = sys.argv[2]
    env = Environment()
    pos_graphs, neg_graphs = env.load_dataset(graph_file, labels_file)
    env.build_domain_info(pos_graphs)
    tmpdir = tempfile.mkdtemp()
    pos_input = os.path.join(tmpdir, "pos_input.txt")
    neg_input = os.path.join(tmpdir, "neg_input.txt")
    pos_sup_file = os.path.join(tmpdir, "gaston_pos.txt")
    neg_sup_file = os.path.join(tmpdir, "gaston_neg.txt")
    write_graphs_to_file(pos_graphs, pos_input)
    write_graphs_to_file(neg_graphs, neg_input)
    pos_min_sup = math.ceil(MIN_SUPPORT_PERCENT * len(pos_graphs))
    neg_min_sup = math.ceil(MIN_SUPPORT_PERCENT * len(neg_graphs))
    run_gaston(pos_input, pos_min_sup, pos_sup_file)
    run_gaston(neg_input, neg_min_sup, neg_sup_file)
    env.load_gaston_supports(pos_sup_file, neg_sup_file)
    all_graphs = pos_graphs + neg_graphs
    labels = [0] * len(pos_graphs) + [1] * len(neg_graphs)
    print(f"Running fast-probe on {len(all_graphs)} graphs ...")
    selected_patterns = fast_probe(all_graphs, labels, env)
    out_folder = "subgraphs_output"
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, "selected_patterns.txt")
    with open(out_file, "w") as f:
        for pat in selected_patterns:
            f.write(f"{pat.get_ccam_code()}\t{pat.calc_discrimination_score():.4f}\n")
    print(f"Selected patterns written to {out_file}")
    import shutil
    shutil.rmtree(tmpdir)

if __name__ == "__main__":
    main()
