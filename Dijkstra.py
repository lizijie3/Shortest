from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.qp = {}
        self.keys = {}
        self.size = 0

    def is_empty(self):
        return self.size == 0

    def extract_min(self):
        if self.size == 0:
            raise Exception('Heap is empty!')
        min_name = self.pq[0]
        self.swap(0, self.size - 1)
        self.pq.pop()
        self.qp[min_name] = 0
        self.keys.pop(min_name)
        self.size -= 1
        self.sink(0)
        return min_name

    def insert(self, name, key):
        self.size += 1
        self.qp[name] = self.size
        self.keys[name] = key
        self.pq.append(name)
        self.swim(self.size - 1)

    def decrease_key(self, name, new_key):
        if self.contains(name):
            self.keys[name] = new_key
            self.sink(self.qp[name])
            self.swim(self.qp[name])
        else:
            raise Exception('No such element exists!')

    def swap(self, i, j):
        temp = self.pq[i]
        self.pq[i] = self.pq[j]
        self.pq[j] = temp

        self.qp[self.pq[i]] = i
        self.qp[self.pq[j]] = j

    def swim(self, k):
        while k > 0 and self.keys[self.pq[k // 2]] > self.keys[self.pq[k]]:
            self.swap(k, k // 2)
            k = k // 2

    def sink(self, k):
        while 2 * k < self.size:
            j = 2 * k
            if j < self.size - 1 and self.keys[self.pq[j]] > self.keys[self.pq[j + 1]]:
                j += 1
            if self.keys[self.pq[k]] > self.keys[self.pq[j]]:
                self.swap(k, j)
                k = j
            else:
                break

    def parent(self, k):
        if k > 0:
            return self.keys[self.pq[k // 2]]
        else:
            return float('inf')

    def left_child(self, k):
        return self.keys[self.pq[2 * k]]

    def right_child(self, k):
        return self.keys[self.pq[2 * k + 1]]

    def contains(self, name):
        return name in self.qp and 1 <= name <= self.size


adj_matrix = np.array([
    [0, 2, 4, 0, 0],
    [2, 0, 1, 7, 0],
    [4, 1, 0, 3, 0],
    [0, 7, 3, 0, 2],
    [0, 0, 0, 2, 0]
])

def dijkstra(adj_matrix, origin, destination):
    num_nodes = len(adj_matrix)

    dist = [float('inf')] * num_nodes
    prev = [None] * num_nodes
    visited = [False] * num_nodes

    dist[origin] = 0

    for _ in range(num_nodes):
        u = min_distance(dist, visited)
        visited[u] = True

        for v in range(num_nodes):
            if not visited[v] and adj_matrix[u][v] > 0 and dist[u] + adj_matrix[u][v] < dist[v]:
                dist[v] = dist[u] + adj_matrix[u][v]
                prev[v] = u

    path = []
    u = destination
    while prev[u] is not None:
        path.insert(0, u)
        u = prev[u]
    path.insert(0, origin)

    return path, dist[destination]

def min_distance(dist, visited):
    min_dist = float('inf')
    min_index = -1
    for v, d in enumerate(dist):
        if not visited[v] and d < min_dist:
            min_dist = d
            min_index = v
    return min_index

@app.route('/shortest_path/<int:origin>/<int:destination>', methods=['GET'])
def shortest_path(origin, destination):
    if origin < 0 or origin >= len(adj_matrix) or destination < 0 or destination >= len(adj_matrix):
        return jsonify({'error': 'Invalid origin or destination'})

    path, distance = dijkstra(adj_matrix, origin, destination)

    # Convert the list of integers to a list of strings
    path = [str(node) for node in path]

    # Ensure distance is a serializable type (e.g., float)
    distance = float(distance)

    return jsonify({'shortest_path': path, 'distance': distance})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)