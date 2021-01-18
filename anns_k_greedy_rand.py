import sys
import random
from tqdm import tqdm
import heapq
from collections import namedtuple
import time

MAXINT = 2 ** 31 - 1

Candidate = namedtuple('Candidate', ['distance', 'id'])


def dist(vec_a, vec_b):
    return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)) ** 0.5


class ANNS:

    def __init__(self, data_path, graph_path, query_path):
        sys.stderr.write("loading data...")
        data = [self.parse_line(line) for line in open(data_path, 'r', encoding='UTF8')]
        sys.stderr.write("loading vector complete.")
        self.D = [d[1] for d in data]  # vectors
        self.num_nodes = len(self.D)

        self.G = [[int(n) for n in line.strip().split()[1:]] for line in open(graph_path, 'r', encoding='UTF8')]
        sys.stderr.write("loading graph complete.")


        self.querys = [ tuple(float(item) for item in line.strip().split(',')) for line in open(query_path, 'r', encoding='UTF8')]
        sys.stderr.write("loading query complete.")

        self.visited = [0] * self.num_nodes
        self.vmark = 2
        self.omark = 1

    def reset_visited(self):
        if self.vmark == MAXINT:
            self.vmark = 2
            self.omark = 1
            for i in range(len(self.visited)):
                self.visited[i] = 0
        else:
            self.vmark += 2
            self.omark += 2

    def parse_line(self, line):
        items = line.strip().split()
        word = items[0]
        vector = tuple(float(item) for item in items[1:])
        return (word, vector)

    # p: start point
    # q: query point
    # l: heap size
    # k: return points
    def k_greedy_search(self, p, q, l, k):

        self.reset_visited()

        dist_pq = dist(self.D[p], self.querys[q])
        S = [Candidate(dist_pq, p)]

        while True:
            i, pi = next(((i, s) for i, s in enumerate(S) if self.visited[s.id] != self.vmark), (None, None))

            if i == None: break

            self.visited[pi.id] = self.vmark
            for n in self.G[pi.id]:
                if self.visited[n] < self.omark:
                    self.visited[n] = self.omark
                    S.append(Candidate(dist(self.D[n], self.querys[q]), n))

            S.sort()
            while len(S) > l:
                S.pop()

        return S[:k]

    def greedy_search(self, p, q):
        min_dist = dist(self.D[p], self.D[q])

        while True:
            if len(self.D[p]) == 0:
                break

            cdist, c = min((dist(self.D[n], self.D[q]), n) for n in self.G[p])

            if cdist < min_dist:
                min_dist = cdist
                p = c
            else:
                break

        return p


data_path = sys.argv[1]
graph_path = sys.argv[2]
query_path = sys.argv[3]
anns = ANNS(data_path, graph_path, query_path)

results = []
k = 30
l = 100

t1 = time.time()
for q in tqdm(range(len(anns.querys))):

    p = random.randint(0, anns.num_nodes - 1)
    res = anns.k_greedy_search(p, q, l, k)
    results.append(res)

t2 = time.time()
sec = int(t2 - t1)
query_per_sec = len(anns.querys) / sec


ans_path = sys.argv[4]
answers = [set(int(n) for n in line.strip().split()[:k]) for line in open(ans_path, 'r', encoding='UTF8')]
recall_sum = 0

for q, res in tqdm(enumerate(results)):

    ans = answers[q]
    true_positive = 0
    for n in res:
        id = n.id
        if id in ans: true_positive +=1

    recall = true_positive / k
    recall_sum += recall

recall_mean = recall_sum / len(anns.querys)

print("data: {}\ngraph: {}\nquery per second: {}\nrecall: {}".format(data_path, graph_path, query_per_sec, recall_mean))
