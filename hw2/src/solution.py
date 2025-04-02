import argparse
import getopt
import heapq
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

# ---------------- Network Representation Classes ---------------- #

# Reverse network (used for generating reverse reachable sets)
class RevNet:
    def __init__(self):
        self.adj_list = {}
    def insert_edge(self, u, v, prob):
        if v not in self.adj_list:
            self.adj_list[v] = []
        self.adj_list[v].append((u, prob))
        if u not in self.adj_list:
            self.adj_list[u] = []
    def neighbors(self, node):
        if node in self.adj_list:
            return self.adj_list[node]
        return []

# Forward network representation
class FwdNet:
    def __init__(self):
        self.adj_list = {}
    def insert_edge(self, u, v, prob):
        if u not in self.adj_list:
            self.adj_list[u] = []
        self.adj_list[u].append((v, prob))
        if v not in self.adj_list:
            self.adj_list[v] = []
    def neighbors(self, node):
        if node in self.adj_list:
            return self.adj_list[node]
        return []

# ---------------- Worker Process for Sampling ---------------- #

class WorkerProcess(mp.Process):
    def __init__(self, in_queue, out_queue, total_nodes, rev_net, proc_id, base_seed, powerFactor, hybridProb):
        super(WorkerProcess, self).__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.samples = []
        self.counter = 0
        self.total_nodes = total_nodes
        self.rev_net = rev_net
        self.proc_id = proc_id
        self.powerFactor = powerFactor      # exponent for degree transformation
        self.hybridProb = hybridProb        # probability to use degree-based sampling
        self.seed = base_seed

        # Precompute candidate nodes and their transformed weights
        self.candidates = list(self.rev_net.adj_list.keys())
        self.weight_list = []
        for node in self.candidates:
            deg = len(self.rev_net.neighbors(node))
            w_val = math.pow(deg, self.powerFactor)
            self.weight_list.append(w_val)

    def choose_candidate(self):
        if random.random() < self.hybridProb:
            return random.choices(self.candidates, weights=self.weight_list, k=1)[0]
        else:
            return random.randint(1, self.total_nodes)

    def run(self):
        # Seed uniquely for each worker.
        random.seed(self.seed + self.proc_id)
        while True:
            task_num = self.in_queue.get()
            while self.counter < task_num:
                if self.hybridProb > 0.0:
                    candidate = self.choose_candidate()
                else:
                    candidate = random.randint(1, self.total_nodes)
                rr_set = compute_rr(candidate, self.rev_net)
                self.samples.append(rr_set)
                self.counter += 1
            self.counter = 0
            self.out_queue.put(self.samples)
            self.samples = []

# ---------------- Worker Process Management ---------------- #

def launch_workers(num_workers, total_nodes, rev_net, base_seed, powerFactor, hybridProb):
    global worker_pool
    for i in range(num_workers):
        q_in = mp.Queue()
        q_out = mp.Queue()
        worker_pool.append(WorkerProcess(q_in, q_out, total_nodes, rev_net, i, base_seed, powerFactor, hybridProb))
        worker_pool[i].start()

def terminate_workers():
    global worker_pool
    for wp in worker_pool:
        wp.terminate()
    worker_pool = []

# ---------------- Reverse Reachable Set Generation ---------------- #

def compute_rr(start, rev_net):
    rr_result = [start]
    frontier = [start]
    seen = {start}
    while frontier:
        new_frontier = []
        for node in frontier:
            for neighbor, prob in rev_net.neighbors(node):
                if neighbor not in seen:
                    if random.random() < prob:
                        seen.add(neighbor)
                        rr_result.append(neighbor)
                        new_frontier.append(neighbor)
        frontier = new_frontier
    return rr_result

# ---------------- Sampling and Node Selection ---------------- #

def compute_log_comb(n, k):
    if k > n or k < 0:
        return 0
    if k > n / 2:
        k = n - k
    total = 0
    for i in range(k):
        total += math.log(n - i)
        total -= math.log(i + 1)
    return total

def generate_samples(epsilon, l_val, base_seed, powerFactor, hybridProb):
    global seed_size, worker_pool
    total_samples = []
    lower_bound = 1
    n_val = total_nodes
    k_val = seed_size
    l_val = l_val * (1 + math.log(2) / math.log(n_val))
    eps_factor = epsilon * math.sqrt(2)
    worker_count = min(mp.cpu_count(), 4)
    launch_workers(worker_count, total_nodes, rev_network, base_seed, powerFactor, hybridProb)
    for i in range(1, int(math.log2(n_val - 1)) + 1):
        start_time_iter = time.time()
        x_val = n_val / (2 ** i)
        lambda_val = ((2 + 2 * eps_factor / 3) *
                      (compute_log_comb(n_val, k_val) + l_val * math.log(n_val) + math.log(math.log2(n_val))) *
                      n_val) / (eps_factor ** 2)
        theta_val = lambda_val / x_val
        tasks_per_worker = math.ceil((theta_val - len(total_samples)) / worker_count)
        for j in range(worker_count):
            worker_pool[j].in_queue.put(tasks_per_worker)
        for wp in worker_pool:
            total_samples += wp.out_queue.get()
        print(f"Iteration {i}: Time for sample generation: {time.time() - start_time_iter:.2f} sec, Total samples: {len(total_samples)}")
        candidate_set, coverage = select_nodes(total_samples, k_val)
        print(f"Current coverage fraction: {coverage:.4f}")
        if n_val * coverage >= (1 + eps_factor) * x_val:
            lower_bound = n_val * coverage / (1 + eps_factor)
            break
    alpha_val = math.sqrt(l_val * math.log(n_val) + math.log(2))
    beta_val = math.sqrt((1 - 1 / math.e) * (compute_log_comb(n_val, k_val) + l_val * math.log(n_val) + math.log(2)))
    lambda_star = 2 * n_val * (((1 - 1 / math.e) * alpha_val + beta_val) ** 2) * (epsilon ** -2)
    theta_final = lambda_star / lower_bound
    extra_tasks = theta_final - len(total_samples)
    if extra_tasks > 0:
        extra_per_worker = math.ceil(extra_tasks / worker_count)
        for j in range(worker_count):
            worker_pool[j].in_queue.put(extra_per_worker)
        for wp in worker_pool:
            total_samples += wp.out_queue.get()
    terminate_workers()
    print(f"Final sample count: {len(total_samples)}")
    final_selection, final_coverage = select_nodes(total_samples, k_val)
    print(f"Chosen {len(final_selection)} seeds with final coverage: {final_coverage:.4f}")
    return final_selection

def select_nodes(rr_sets, k_val):
    mapping = {}
    for idx, rr in enumerate(rr_sets):
        for node in rr:
            if node not in mapping:
                mapping[node] = set()
            mapping[node].add(idx)
    covered_sets = set()
    cache_vals = {node: len(indices) for node, indices in mapping.items()}
    heap = [(-cache_vals[node], node) for node in mapping]
    heapq.heapify(heap)
    chosen_set = set()
    while len(chosen_set) < k_val and heap:
        neg_val, candidate = heapq.heappop(heap)
        current_val = len(mapping[candidate] - covered_sets)
        if current_val != -neg_val:
            cache_vals[candidate] = current_val
            heapq.heappush(heap, (-current_val, candidate))
            continue
        if current_val == 0:
            break
        chosen_set.add(candidate)
        covered_sets |= mapping[candidate]
        if len(covered_sets) == len(rr_sets):
            break
    return chosen_set, len(covered_sets) / len(rr_sets)

# ---------------- File Reading Function ---------------- #

def load_network(file_path):
    global total_nodes, total_edges, rev_network
    lines = open(file_path, 'r').readlines()
    nodes = set()
    edge_list = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            u = int(parts[0])
            v = int(parts[1])
            prob = float(parts[2])
            edge_list.append((u, v, prob))
            nodes.add(u)
            nodes.add(v)
        except ValueError:
            print(f"Warning: Skipping invalid line: {line.strip()}")
    total_nodes = max(nodes)
    total_edges = len(edge_list)
    print(f"Loaded network with {total_nodes} nodes and {total_edges} edges")
    rev_network = RevNet()
    fwd_network = FwdNet()
    for (u, v, p) in edge_list:
        rev_network.insert_edge(u, v, p)
        fwd_network.insert_edge(u, v, p)
    return fwd_network

# ---------------- Main Function ---------------- #

if __name__ == "__main__":
    start_time = time.time()
    global total_nodes, seed_size, rev_network
    total_nodes = 0
    seed_size = 0
    rev_network = RevNet()
    worker_pool = []
    parser = argparse.ArgumentParser(description="COL761 Homework 2 Solution")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input graph file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file for seed set")
    parser.add_argument("-k", "--seeds", type=int, default=50, help="Number of seeds")
    parser.add_argument("-v", "--verify", type=int, default=100, help="Number of verification instances")
    args = parser.parse_args()
    
    in_file = args.input
    out_file = args.output
    seed_size = args.seeds
    verify_count = args.verify
    
    load_network(in_file)
    
    total_deg = sum(len(rev_network.adj_list[node]) for node in rev_network.adj_list)
    avg_deg = total_deg / len(rev_network.adj_list)
    print(f"Average degree of the network: {avg_deg}")
    
    if avg_deg < 7:
        eps_val = 0.1
        lambda_val = 50
        powerFactor = 0.1
        base_random = 1018
        hybridProb = 0.0
    else:
        eps_val = 0.1
        lambda_val = 50
        powerFactor = 0.1
        base_random = 1028
        hybridProb = 1.0

    random.seed(base_random)
    
    print(f"\n=== Running with epsilon={eps_val}, lambda_val={lambda_val}, seed={base_random}, powerFactor={powerFactor}, hybridProb={hybridProb} on {in_file} ===")
    selected_seeds = generate_samples(eps_val, lambda_val, base_random, powerFactor, hybridProb)
    
    with open(out_file, 'w') as fout:
        for s in selected_seeds:
            fout.write(f"{s}\n")
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
