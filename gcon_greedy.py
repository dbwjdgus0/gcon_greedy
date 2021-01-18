import sys
import random
from typing import List
from tqdm import tqdm
import argparse


def euclidean_distance(vec_a, vec_b):
    s = 0
    for a, b in zip(vec_a, vec_b):
        s += (a - b) ** 2

    return s ** 0.5

    # return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)) ** 0.5


# q: query vector
# s: start point
# adj: graph in adjacency list format


class GConGreedy:

    def __init__(self, data, initial_adj=None, gamma=0.5, alpha=None, beta=1, occl=False):
        self.gamma = gamma
        self.alpha = gamma if alpha is None else alpha
        self.beta = beta
        self.occl = occl
        self.vectors = [v for w, v in data]
        if initial_adj == None:
            self.adj = [[] for _ in self.vectors]
            self.num_edges = 0
            self.prefix = ""
        else:
            self.adj = initial_adj
            self.prefix = "init"
            self.num_edges = sum(len(x) for x in initial_adj)

        self.num_added_by_ls = float('inf')
        self.mindists = [float("inf")] * len(self.vectors)

        self.construct()

    def construct(self):

        dvc_ratio = self.alpha

        for it in range(10000):
            num_succ = 0
            for q, qvec in enumerate(tqdm(self.vectors, desc=f"grd-{self.gamma}-{it}-{dvc_ratio:.4f}-build")):

                if self.occl:
                    a = self.greedy_search_with_occl(qvec, random.randint(0, len(self.vectors) - 1))
                else:
                    a = self.greedy_search(qvec, random.randint(0, len(self.vectors) - 1))

                if a != q:
                    self.adj[a].append([q, 1, 2])
                    self.num_edges += 1
                    dist_aq = euclidean_distance(self.vectors[a], self.vectors[q])
                    if dist_aq < self.mindists[a]:
                        self.mindists[a] = dist_aq
                else:
                    num_succ += 1
            print(f"{it}-1:\t{self.num_edges}\t{num_succ}")

            self.local_search()

            for arr in self.adj:
                arr.sort(key=lambda x: x[1], reverse=True)
                while len(arr) > 0 and arr[-1][1] == 0:
                    arr.pop()
                    self.num_edges -= 1

            self.decrease_visit_count(dvc_ratio)
            dvc_ratio = max(self.gamma, dvc_ratio * self.beta)

            if it % 10 == 0:
                self.save_adj_as_file(f"adj_iter_{self.alpha}_{self.gamma}_{self.beta}_{self.prefix}_{it}")

            num_succ_real = 0
            for q, qvec in enumerate(tqdm(self.vectors, desc=f"grd-{self.gamma}-{it}-test")):
                a = self.greedy_search(qvec, random.randint(0, len(self.vectors) - 1), False)
                if a == q:
                    num_succ_real += 1

            print(f"{it}-2:\t{self.num_edges}\t{num_succ_real}")
            sys.stdout.flush()

    def local_search(self):

        if self.num_added_by_ls < 1000:
            return

        self.num_added_by_ls = 0
        for n, nvec in enumerate(tqdm(self.vectors, desc=f"localsearch")):
            for u, _, isnew_u in self.adj[n]:
                for v, _, isnew_v in self.adj[u]:
                    if isnew_u == 0 and isnew_v == 0:
                        continue
                    
                    if (n != v) and (v not in (x[0] for x in self.adj[n])):
                        dist_nv = euclidean_distance(self.vectors[n], self.vectors[v])
                        if (dist_nv < self.mindists[n]):
                            self.num_added_by_ls += 1
                            self.adj[n].append([v, 1, 2])
                            self.mindists[n] = dist_nv
                            self.num_edges += 1

        for n, nvec in enumerate(self.vectors):
            for u in self.adj[n]:
                u[-1] = max(0, u[-1]-1)

        print(f"ls:\t{self.num_edges}\t{self.num_added_by_ls}")

    def decrease_visit_count(self, k):
        for arr in self.adj:
            for pair in arr:
                pair[1] = max(pair[1] - k, 0)

    def save_adj_as_file(self, path):
        with open(path, 'w') as f:
            for i, arr in enumerate(self.adj):
                f.write(f"{i}")
                for a, _ in arr:
                    f.write(f" {a}")
                f.write("\n")

    def greedy_search_with_occl(self, q, s):
        dist_sq = euclidean_distance(self.vectors[s], q)

        while True:
            if len(self.adj[s]) == 0:
                break

            min_dist_cq = dist_sq
            min_sc = None

            for sc in self.adj[s]:
                c = sc[0]
                dist_sc = euclidean_distance(self.vectors[s], self.vectors[c])
                dist_cq = euclidean_distance(self.vectors[c], q)
                if dist_sc < dist_sq:
                    sc[1] += 1
                    if dist_cq < min_dist_cq:  # occlusion rule
                        min_dist_cq = dist_cq
                        min_sc = sc

            if min_sc is None:
                break

            dist_sq = min_dist_cq
            sc[1] += 1
            s = sc[0]

        return s

    def greedy_search(self, q, s, vc=True):
        min_dist = euclidean_distance(self.vectors[s], q)

        while True:
            if len(self.adj[s]) == 0:
                break

            dist, c = min((euclidean_distance(self.vectors[u[0]], q), u) for u in self.adj[s])

            if dist < min_dist:
                min_dist = dist
                if vc: c[1] += 1
                s = c[0]
            else:
                break

        return s


def parse_line(line):
    items = line.strip().split()
    word = items[0]
    vector = tuple(float(item) for item in items[1:])
    return (word, vector)


def parse_graph(gpath, num_nodes):
    initial_graph = [[] for _ in range(num_nodes)]

    for line in open(gpath, 'r'):
        items = (int(w) for w in line.strip().split())
        u = next(items)
        initial_graph[u] = [[v, 1] for v in items]

    return initial_graph


parser = argparse.ArgumentParser(prog='gcon_greedy')
parser.add_argument('-p', metavar='PRR', help='pheromone reduction ratio', required=True, type=float)
parser.add_argument('-b', metavar='BETA', type=float, default=1.0)
parser.add_argument('-a', metavar='ALPHA', type=float, default=None)
parser.add_argument('--occl', help='occlusion rule', action='store_true')
parser.add_argument('-i', metavar='INITIAL_GRAPH', help='initial graph')
args = parser.parse_args(sys.argv[1:])

if args.a is None:
    args.a = args.p

sys.stderr.write(f"{args}\n")

data = [parse_line(line) for line in sys.stdin]

initial_graph = parse_graph(args.i, len(data)) if args.i else None
gcg = GConGreedy(data, initial_graph, gamma=args.p, alpha=args.a, beta=args.b, occl=args.occl)

