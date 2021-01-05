import sys
import random
from typing import List
from tqdm import tqdm


def euclidean_distance(vec_a, vec_b):
    return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)) ** 0.5


# q: query vector
# s: start point
# adj: graph in adjacency list format

class aknng:

    def __init__(self, data):
        self.vectors = [v for w, v in data]
        self.adj = [[] for _ in self.vectors]
        self.num_edges = 0

        self.successed = []

        self.construct()



    def construct(self):

        for it in range(1001):
            acc_num_succ = 0
            for _ in range(1):
                num_succ = 0
                for q, qvec in enumerate(self.vectors):
                    if q in self.successed:
                        continue
                    a = self.greedy_search(qvec, random.randint(0, len(self.vectors) - 1))
                    if a != q:
                        self.adj[a].append([q, 1])
                        self.successed.append(q)
                        self.num_edges += 1
                    else:
                        num_succ += 1
                print(f"{it}-1: {self.num_edges}\t{num_succ}")
                acc_num_succ += num_succ

            for arr in self.adj:
                arr.sort(key=lambda x: x[1], reverse=True)
                while len(arr) > 0 and arr[-1][1] == 0:
                    arr.pop()
                    self.num_edges -= 1

            # self.zero_visit_count()
            self.decrease_visit_count(0.5)
            if it % 10 == 0:
                self.save_adj_as_file(f"adj_iter_{it}")
            print(f"{it}-2: {self.num_edges}\t{acc_num_succ}")

    def zero_visit_count(self):
        for arr in self.adj:
            for pair in arr:
                pair[1] = 0

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

    def greedy_search(self, q, s):
        min_dist = euclidean_distance(self.vectors[s], q)

        while True:
            if len(self.adj[s]) == 0:
                break

            dist, c = min((euclidean_distance(self.vectors[u[0]], q), u) for u in self.adj[s])

            if dist < min_dist:
                min_dist = dist
                c[1] += 1
                s = c[0]
            else:
                break

        return s

def parse_line(line):
    items = line.strip().split()
    word = items[0]
    vector = tuple(float(item) for item in items[1:])
    return (word, vector)


data = [parse_line(line) for line in sys.stdin]

ak = aknng(data)
