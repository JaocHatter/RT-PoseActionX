import numpy as np
from models.graphs.tools import get_hierarchical_graph, get_edgeset

num_node = 25

class Graph:
    def __init__(self, CoM=21, labeling_mode='spatial'):
        self.num_node = num_node
        self.CoM = CoM
        self.A = self.get_adjacency_matrix(labeling_mode)
        

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_hierarchical_graph(num_node, get_edgeset(dataset='NTU', CoM=self.CoM)) # L, 3, 25, 25
        else:
            raise ValueError()
        return A, self.CoM
