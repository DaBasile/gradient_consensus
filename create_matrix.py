from random import randint

import networkx as nx
import numpy as np


def createAdjM(world_d):
    g = nx.DiGraph()
    for i in range(0, world_d):
        g.add_node(i)
    for j in range(0, world_d):
        if j == world_d-1:
            g.add_edge(j, 0)
        else:
            g.add_edge(j, j + 1)
    adj = nx.adjacency_matrix(g)
    return g


def createR(d):
    r = np.zeros((d, 1))
    for i in range(0, d):
        r[i] = randint(0, 1)
    return r


def createQ(d):
    q = np.zeros((d, d))
    eye = np.identity(d)
    for i in range(0, d):
        for j in range(0, d):
            q[i][j] = randint(0, 2)
    a = np.add(q, np.transpose(q))
    a = np.add(a, np.dot(d, eye))
    return a


