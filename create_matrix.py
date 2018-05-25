from random import randint
from warnings import warn
import networkx as nx
import numpy as np


def createAdjM(world_d, n_edges=None, phi=None):

    if n_edges is None:
        n_edges = 1

    if n_edges > world_d or n_edges == 0:
        warn("Invalid n_edges entered... Setting to 1")
        n_edges = 1

    if phi is None:
        phi = 0

    if phi > world_d - 1:
        warn("Invalid phi entered... Setting to 0")
        phi = 0

    g = nx.DiGraph()

    for i in range(0, world_d):
        g.add_node(i)

    for j in range(0, world_d):

        for k in range(0, n_edges):

            e = j + k + phi + 1

            if e >= world_d:
                e = e - world_d

                if e == j:
                    e = e + 1

                    if e >= world_d:
                        e = e - world_d

                g.add_edge(j, e)

            else:
                g.add_edge(j, e)

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
