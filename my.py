from mpi4py import MPI
import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import random

import create_matrix as cm
import calculate_function as cf

""" Define world parameter, these have been got from mpi system """
world = MPI.COMM_WORLD
world_size = world.Get_size()
rank = world.Get_rank()
name = MPI.Get_processor_name()
""" Define initial value of our test variable """
max_iterations = 50
d = 2
alpha = 0.001
matrix = cm.createAdjM(world_size)


def send_data(data):

    for destination in matrix.successors(rank):
        # BLOCCANTE
        world.send(data, dest=destination)
    sincronizza()


def get_data():
    data_recv = []

    for receive in matrix.predecessors(rank):
        data_recv.append(world.recv(source=receive))
        sincronizza()

    return data_recv


def sincronizza():
    sys.stdout.flush()
    world.Barrier()


"""
def trova_media(iterazioni):
    # "distribute initial constraint" (fasullo) @DA FARE

    for i in range(0, world_size):
        if i == rank:
            data = i


            sincronizza()
            if rank == 0:
                print("numero iterazioni :", iterazioni)

            sincronizza()

            for i in range(0, iterazioni):
                # print("*** Iterazione n:", i, " ***")
                #ricevuto = exchange_with_neighbors(data)

                # print(ricevuto[0] , " da ", rank)
                data = (ricevuto[0] + data) / (matrix.in_degree(rank) + 1)
                #print(data)
                sincronizza()

            print("\tconsenso: ", np.round(data, 2))

"""


def calculate_consensus(x_0, data_q, data_r):
    x_x = [x_0]
    sincronizza()

    for i in range(0, max_iterations):
        tmp = 0
        sincronizza()

        for j in range(0, world_size):
            send_data(x_x[:][0])
            data = get_data()
            u_i = np.zeros((d, 1))

            for h in range(0, world_size):
                u_i = np.sum([u_i, data], axis=0)

            # print(np.dot(alpha, cf.gradientF(x_x[:][i], data_q, data_r)))
            u_i = np.subtract(u_i, np.dot(alpha, cf.gradientF(x_x[:][i], data_q, data_r)))

            # print("time ", i, "\tvalue =  ", u_i)
            x_x.append(u_i)
            tmp = tmp + cf.quadraticF(x_x[:][i], data_q, data_r)
            sincronizza()

            #if rank == 0:
             #   print(x_x[:][i+1])

        sincronizza()


if __name__ == '__main__':

    sys.stdout.write("Hello, World! I am process %d of %d on %s.\n" % (rank, world_size, name))
    sincronizza()
    for i in range(0, world_size):
        if i == rank:
            x_0 = np.array([[i], [i+1]])
            data_q = cm.createQ(d)
            data_r = cm.createR(d)

    if rank == 0:
        print("\n\n\t\tinizio test...\n")

    sincronizza()
    calculate_consensus(x_0, data_q, data_r)
    sincronizza()

    if rank == 0:
        print("\n\n\t\tfine test...\n")
