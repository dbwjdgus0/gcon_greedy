import sys
import random
from typing import List
from tqdm import tqdm


def euclidean_distance(vec_a, vec_b):
    return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)) ** 0.5


# q: query vector
# s: start point
# adj: graph in adjacency list format
# node: [nid, visit_count, dist, old/new]

class GConGreedy:

    def __init__(self, data, initial_adj=None):
        self.vectors = [v for w, v in data]

        self.sample_rate = 1.0
        self.K = 5
        self.scan_sum = 0
        self.reverses = [[]for _ in self.vectors]

        if initial_adj == None:
            self.adj = [[] for _ in self.vectors]
            self.num_edges = 0
        else:
            self.adj = initial_adj
            self.num_edges = sum(len(x) for x in initial_adj)


        for it in range(10):

            self.scan = 0
            self.gcon(it)
            self.nndes(it)
            print(self.scan)
            self.scan_sum += self.scan

        n = len(self.vectors)
        print("scan sum: {}".format(self.scan_sum))

    def nndes(self, it):

        suc = 0
        old = []
        new = []
        old_rev = []
        new_rev = []

        for id, neighbors in enumerate(self.adj):

            o = [node[0] for node in neighbors if node[-1] == 'o']
            n = [node[0] for node in neighbors if node[-1] == 'n']

            o_rev = [node[0] for node in self.reverses[id] if node[-1] == 'o']
            n_rev = [node[0] for node in self.reverses[id] if node[-1] == 'n']

            for i in range(len(self.adj[id])):
                self.adj[id][i][-1] = 'o'
            for i in range(len(self.reverses[id])):
                self.reverses[id][i][-1] = 'o'


            old.append(o)
            new.append(n)
            old_rev.append(o_rev)
            new_rev.append(n_rev)



        for q, qvec in enumerate(self.vectors):

            old[q] = list(set(old[q] + old_rev[q]))
            new[q] = list(set(new[q] + new_rev[q]))

            for i, v in enumerate(new[q]):
                candidates = new[q][i+1:] + old[q]

                for u in candidates:
                    l = euclidean_distance(self.vectors[u], self.vectors[v])
                    self.scan += 1
                    suc += self.update(v, u, l)
                    suc += self.update(u, v, l)

        print(f"{it}-3: {self.num_edges}\t{suc}")
        #if it % 10 == 0:
        self.save_adj_as_file(f"adj_iter_{it}")

    def sample(self, arr, n):

        if arr == None:
            return []
        else:
            n = min(n, len(arr))
            random.shuffle(arr)
            return arr[:n]

    def reverse(self, arr):

        rev = [[] for _ in arr]

        for v, neighbors in enumerate(arr):
            for u in neighbors:
                rev[u].append(v)

        return rev

    def update(self, v, u, l):

        if u in [node[0] for node in self.adj[v]]:
            return 1
        if u == v:
            return 0

        if len(self.adj[v]) < self.K:
            self.adj[v].append([u, 1, l, 'n'])
            self.reverses[u].append([v,'n'])
            self.num_edges += 1
        else:
            self.adj[v].sort(key=lambda x: x[2])
            if l < self.adj[v][-1][2]:

                if self.adj[v][-1][-1] == 'n':
                    self.reverses[self.adj[v][-1][0]].remove([v, 'n'])
                else:
                    self.reverses[self.adj[v][-1][0]].remove([v, 'o'])

                self.adj[v].pop()
                self.adj[v].append([u, 1, l, 'n'])
                self.reverses[u].append([v, 'n'])
        return 0

    def gcon(self, it):
        acc_num_succ = 0

        for _ in range(1):
            num_succ = 0
            for q, qvec in enumerate(self.vectors):
                a = self.greedy_search(qvec, random.randint(0, len(self.vectors) - 1))
                if a != q:
                    dis = euclidean_distance(self.vectors[a], qvec)
                    self.update(a, q, dis)
                else:
                    num_succ += 1
            print(f"{it}-1: {self.num_edges}\t{num_succ}")
            acc_num_succ += num_succ

        for v, arr in enumerate(self.adj):
            arr.sort(key=lambda x: x[1], reverse=True)
            while len(arr) > 0 and arr[-1][1] == 0:

                if arr[-1][-1] == 'n':
                    self.reverses[arr[-1][0]].remove([v, 'n'])
                else:
                    self.reverses[arr[-1][0]].remove([v, 'o'])
                arr.pop()
                self.num_edges -= 1

        # self.zero_visit_count()
        self.decrease_visit_count(0.4)
        print(f"{it}-2: {self.num_edges}\t{acc_num_succ}")


        sys.stdout.flush()

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
                for each in arr:
                    f.write(f" {each[0]}")
                f.write("\n")

    def greedy_search(self, q, s):
        min_dist = euclidean_distance(self.vectors[s], q)
        self.scan += 1
        while True:
            if len(self.adj[s]) == 0:
                break

            dist, c = min((euclidean_distance(self.vectors[u[0]], q), u) for u in self.adj[s])
            self.scan += len(self.adj[s])

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


def parse_nn(gpath, num_nodes, k):
    initial_graph = [[] for _ in range(num_nodes)]

    for line in open(gpath, 'r'):
        items = (int(w) for w in line.strip().split())
        u = next(items)
        for i in range(k):
            v = next(items)
            initial_graph[u].append([v, 1])

    return initial_graph

if __name__ == '__main__':

    #d = open('glove50d_test.txt', 'r')
    d = open('glove.6B.50d.txt', 'r')
    data = [parse_line(line) for line in d]
    #data = [parse_line(line) for line in sys.stdin]
    # if len(sys.argv) >= 2:
    #     sys.stderr.write("init!\n")
    #     initial_graph = parse_nn(sys.argv[1], len(data), 5)
    #     sys.stderr.write("init~~~!\n")
    #     gcg = GConGreedy(data, initial_graph)
    # else:
    gcg = GConGreedy(data)