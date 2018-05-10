from mpi4py import MPI
import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import random

import create_matrix as cm
import calculate_function as cf

max_iterations = 50
world = MPI.COMM_WORLD
world_size = world.Get_size()
rank = world.Get_rank()
name = MPI.Get_processor_name()


def load_adj_matrix():
    input = cm.createAdjM(world_size)

    #print(input.todense())
    return input

# TODO MATRICE DEI PESI @DA FARE


def out_neighbors_of_node(x):
    # lista di riceventi di riga/nodo x

    matrix = load_adj_matrix()
    list_out_neighbors = []
    # shape[1], numero colonne della matrice
    for i in range(0, matrix.shape[1]):
        if matrix[x, i] == 1:
            list_out_neighbors.append(i)
    return list_out_neighbors


def in_neighbors_of_node(x):
    # lista di chi invia alla colonna/nodo x

    matrix = load_adj_matrix()
    list_in_neighbors = []
    # shape[0], numero righe della matrice
    for i in range(0, matrix.shape[0]):
        if matrix[i, x] == 1:
            list_in_neighbors.append(i)
            print(list_in_neighbors)
    return list_in_neighbors


def out_degree_of_node(x):
    return len(out_neighbors_of_node(x))


def in_degree_of_node(x):
    return len(in_neighbors_of_node(x))


def exchange_with_neighbors(data):
    # invia a tutti i vicini collegati
    out_neighbors_list = out_neighbors_of_node(rank)
    out_degree_of_actual_node = out_degree_of_node(rank)
    # #   print(    "out of process", rank, "is made of: ", out_neighbors_list, ". Total:",
    # out_degree_of_actual_node, "element(s)")
    for j in range(0, out_degree_of_actual_node):
        # BLOCCANTE

        world.send(data, dest=out_neighbors_list[j])

    # print("rank: ", rank, " ha inviato ", data, " (data) ", " a: ", out_neighbors_list[j], "\n")

    # sincronizza()

    # ricevi
    data_recv = []
    in_neighbors_list = in_neighbors_of_node(rank)
    in_neighbors_list = in_neighbors_of_node(rank)
    in_degree_of_actual_node = in_degree_of_node(rank)
    # print("in of process", rank, "is made of: ", in_neighbors_list, ". Total:", in_degree_of_actual_node,
    # "element(s)")

    for j in range(0, in_degree_of_actual_node):
        # print("rank: ", rank, " cerca di ricevere da: ", in_neighbors_list[j])

        # BLOCCANTE
        data_recv.append(world.recv(source=in_neighbors_list[j]))

    # print("rank: ", rank, " ha ricevuto ", data_recv, "da: ", in_neighbors_list[j], "\n")

    sincronizza()
    return data_recv


# def distribute_initial_constraints():
#     if (rank == 0):
#         data = 4
#     if (rank == 1):
#         data = 5
#     if (rank == 2):
#         data = 6
#     if (rank == 3):
#         data = 7


def sincronizza():
    sys.stdout.flush()
    world.Barrier()


def trova_media(iterazioni):
    # "distribute initial constraint" (fasullo) @DA FARE

    data = 0

    for i in range(0, world_size):
        if i == rank:
            data = 1 * random(5)
            dataQ = cm.createQ(1)
            dataR = cm.createR(1)
    """        
    if rank == 0:
        data = 1
        dataQ = cm.createQ(1)
        dataR = cm.createR(1)
    if rank == 1:
        data = 3
        dataQ = cm.createQ(1)
        dataR = cm.createR(1)
    if rank == 2:
        data = 5
        dataQ = cm.createQ(1)
        dataR = cm.createR(1)
    if rank == 3:
        data = 4
        dataQ = cm.createQ(1)
        dataR = cm.createR(1)
    """
    print("data iniziale: ", dataQ, dataR)

    sincronizza()
    if rank == 0:
        print("numero iterazioni :", iterazioni)

    sincronizza()

    for i in range(0, iterazioni):
        # print("*** Iterazione n:", i, " ***")
        ricevuto = exchange_with_neighbors(data)

        # print(ricevuto[0] , " da ", rank)

        data = (ricevuto[0] + data) / (in_degree_of_node(rank) + 1)
        # print(data)
        sincronizza()

    print("\tconsenso: ", np.round(data, 2))


sys.stdout.write("Hello, World! I am process %d of %d on %s.\n" % (rank, world_size, name))
sincronizza()


if __name__ == '__main__':

    matrice = load_adj_matrix()

    if matrice.shape[0] != world_size:
        print("NUMERO DI NODI DIVERSO DAL NUMERO DI NODI MATRICE")
        sys.exit()

    if rank == 0:
        print("MATRICE DI ADIACENZA")
        print(matrice)
    sincronizza()

    if rank == 0:
        print("\n\n\t\tprimo test...\n")
    sincronizza()
    trova_media(5)
    sincronizza()
    if rank == 0:
        print("\n\n\t\taltra prova...\n")

    sincronizza()
    trova_media(20)
