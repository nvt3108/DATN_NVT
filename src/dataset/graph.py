import logging
import numpy as np

class Graph():
    def __init__(self, dataset, max_hop=3, dilation=1):
        self.dataset = dataset.split('-')[0]
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        num_node = 23
        neighbor_link = [(20,18), (18,16), (20,16), (22,16), (16,14), (14,12), (12,11),
                         (19,17), (17,15), (19,15), (21,15), (15,13), (13,11), (11,10), (10,9),
                         (9,0), (8,6), (6,5), (5,4), (4,0), (7,3), (3,2), (2,1), (1,0)]
        connect_joint = np.array([0,0,1,2,0,4,5,3,6,0,9,10,11,11,12,13,14,15,16,17,18,15,16])
                                # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
        parts = [
            np.array([11, 13, 15, 17, 19, 21]),  # left_arm
            np.array([12, 14, 16, 18, 20, 22]),  # right_arm
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])   # head
        ]
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        tranfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(tranfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        vaild_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in vaild_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(vaild_hop), self.num_node, self.num_node))
        for i, hop in enumerate(vaild_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i,i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD