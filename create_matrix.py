from random import randint

import networkx as nx
import numpy as np


def createAdjM(d):
    G = nx.Graph()
    for i in range(0, d):
        G.add_node(i)

    print(G.number_of_nodes())
    Adj = nx.adjacency_matrix(G)

    return Adj


def createR(d):
    r = np.zeros(d)
    for x in range(0, d):
        r[x] = randint(0, 1)
    return r


def createQ(d):
    Adj = np.zeros((d, d))
    A = np.zeros((d, d))
    eye = np.identity(d)
    for x in range(0, d):
        for y in range(0, d):
            Adj[x][y] = randint(0, 2)
            if x == y:
                eye[x][y] = 1
            else:
                eye[x][y] = 0

    for i in range(d):
        for j in range(d):
            A = np.add(Adj, np.transpose(Adj))

            A = np.add(A, np.dot(d, eye))

    return A


if __name__ == '__main__':

    x = createQ(5)

    y = createR(5)

    z = createAdjM(3)

    print(x)
    print(y)
    #print(z.todense())
