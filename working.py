from mpi4py import MPI
import create_matrix as cm
import numpy as np
import matplotlib.pyplot as plt


def quadraticF(x, Q, r):
    f = 1 / 2 * np.dot(np.dot(np.transpose(x), Q), x) + np.dot(np.transpose(r), x)
    return f


def gradientF(x, Q, r):
    df = np.dot(np.transpose(Q) + Q, x) + np.transpose(r)
    return df


""" Define world parameter, these have been got from mpi system """
world = MPI.COMM_WORLD
agents_number = world.Get_size()
rank = world.Get_rank()

""" Define variables """
MAX_ITERATIONS = 10000
dimensions = 4
epsilon = 0.000001
alpha = 0.0001

adj = cm.createAdjM(agents_number)
Q = cm.createQ(dimensions)
r = cm.createR(dimensions)
x0 = [5 for j in range(0, dimensions)]

XX = np.zeros([MAX_ITERATIONS, dimensions])
XX[0] = x0

num_of_neighbors = 0
for in_neighbors in adj.predecessors(rank):
    num_of_neighbors = num_of_neighbors + 1
weight = 1 / (num_of_neighbors + 1)  # 1 is for self-loop

world.Barrier()

tmp = []

for tt in range(1, MAX_ITERATIONS-1):

    # Local variable to store current state
    u_i = np.zeros(dimensions)

    # Send the state to neighbors
    for node in adj.successors(rank):
        world.send(XX[tt], dest=node)

    # Update with state of all nodes before me
    for node in adj.predecessors(rank):
        u_i = u_i + world.recv(source=node) * weight

    # Update with my previous state
    u_i = u_i + XX[tt] * weight

    # Go in the opposite direction with respect to the gradient
    u_i = u_i - alpha * gradientF(XX[tt], Q, r)
    # Store  my new state
    XX[tt+1] = u_i

    tmp.append(quadraticF(XX[tt], Q, r))

    # synchronise
    world.Barrier()

print(XX[len(XX)-1])

if rank != 0:
    world.send(tmp, dest=0)

_sum = np.zeros(MAX_ITERATIONS)
if rank == 0:
    function_values = [tmp]

    for agent in range(1, agents_number):
        function_values.append(world.recv(source=agent))

    for agent in range(0, agents_number - 1):
        for t in range(0, MAX_ITERATIONS-2):
            _sum[t] = _sum[t] + function_values[agent][t]

if rank == 0:
    plt.plot(range(0, MAX_ITERATIONS-3), _sum[0:MAX_ITERATIONS-3])
    plt.title("$\sum_{i=0}^" + str(agents_number) + " f_i$")
    plt.show()
