from mpi4py import MPI
import create_matrix as cm
import numpy as np
import matplotlib.pyplot as plt


CONSTANT_TO_SUBTRACT = 15


def loss(all_theta, category_count):
    # _index = 0
    the_sum = 0
    for index in range(0, len(personal_dataset)):
        denominator = 0
        for theta in all_theta:
            denominator = denominator + np.exp(np.dot(theta, personal_dataset[index][0:4]) - CONSTANT_TO_SUBTRACT)
        for category in range(0, category_count):
            if category == personal_dataset[index][4]:
                _exp = np.exp(np.dot(all_theta[category], personal_dataset[index][:4]) - CONSTANT_TO_SUBTRACT)
                _log = np.log(np.divide(_exp, denominator))
                the_sum = the_sum - _log
    return the_sum


def gradientF(all_theta, category_count):
    thetas = np.zeros(dimensions)
    for index in range(0, len(personal_dataset)):
        denominator = 0
        for theta in all_theta:
            denominator = denominator + np.exp(np.dot(theta, personal_dataset[index][:4]) - CONSTANT_TO_SUBTRACT)
        for category in range(0, category_count):
            coeff = 0
            if category == personal_dataset[index][4]:
                coeff = 1
            _exp = np.exp(np.dot(all_theta[category], personal_dataset[index][:4]) - CONSTANT_TO_SUBTRACT)
            coeff = coeff - np.divide(_exp, denominator)
            thetas[category] = thetas[category] - np.multiply(personal_dataset[index][:4], coeff)
    return thetas


def normalize_dataset(dataset):
    for row in dataset:
        to_scale = row[:4]
        _max = max(to_scale)
        _min = min(to_scale)
        row[:4] = np.divide(np.subtract(to_scale, _min), np.subtract(_max, _min))
    return dataset


# --oversubscribe -n 6
dataset = np.loadtxt('iris_training.txt', delimiter=';', dtype=float)
#dataset = normalize_dataset(dataset)

""" Define world parameter, these have been got from mpi system """
world = MPI.COMM_WORLD
agents_number = world.Get_size()
rank = world.Get_rank()

""" Define variables """
MAX_ITERATIONS = 10000
dimensions = [4, 4]
epsilon = 0.000001

# Assign dataset to each agent
dataset_portion = len(dataset) / agents_number
start_dataset = rank * dataset_portion
end_dataset = (rank  * dataset_portion) + dataset_portion
start_dataset = int(start_dataset)
end_dataset = int(end_dataset)
# if (end_dataset >= len(dataset)):
    # end_dataset = len(dataset) + 1
personal_dataset = dataset[start_dataset:end_dataset]
print(personal_dataset)
# TODO il dataset va sistemato per prendere tutti le rows nel caso di num agenti non divisibile per 30

print("agent ", rank, " got ", len(personal_dataset), " rows of dataset")

adj = cm.createAdjM(agents_number)

x0 = np.ones(dimensions)

XX = np.zeros([MAX_ITERATIONS, *dimensions])
losses = np.zeros(MAX_ITERATIONS)
XX[0] = x0

num_of_neighbors = 0
for in_neighbors in adj.predecessors(rank):
    num_of_neighbors = num_of_neighbors + 1
weight = 1 / (num_of_neighbors + 1)  # 1 is for self-loop

world.Barrier()

for tt in range(1, MAX_ITERATIONS-1):

    alpha = 0.01 * (1 / tt)**0.4

    # Update with my previous state
    u_i = np.multiply(XX[tt-1], weight)

    # Send the state to neighbors
    for node in adj.successors(rank):
        world.send(XX[tt-1], dest=node)

    # Update with state of all nodes before me
    for node in adj.predecessors(rank):
        u_i = u_i + world.recv(source=node) * weight

    # Go in the opposite direction with respect to the gradient
    grad = np.multiply(alpha, gradientF(XX[tt-1], 4))

    for i in range(0, dimensions[0]):
        u_i[i] = np.subtract(u_i[i], grad[i])
    # Store  my new state
    XX[tt] = u_i

    losses[tt] = loss(XX[tt], 4)

    # synchronise
    world.Barrier()

#print(XX[len(XX)-3])

if rank != 0:
    world.send(losses, dest=0)

if rank == 0:

    # Take the losses from all the other agents and sum
    # We now have the overall loss given from the cost function
    for i in range(1, agents_number):
        agent_loss = world.recv(source=i)
        losses = np.add(losses, agent_loss)

    # TODO Dovremmo plottare la somma delle perdite!
    plt.ion()
    plt.show()
    plt.plot(range(0, MAX_ITERATIONS-3), losses[0:MAX_ITERATIONS-3])
    plt.title("$\sum_{i=0}^" + str(agents_number) + " f_i$")
    plt.draw()
    plt.pause(.001)


if rank == 0:
    to_find = np.loadtxt('iris_training_complete.txt', delimiter=';', dtype=float)
    #to_find = normalize_dataset(to_find)
    wrong_answers = 0
    for _set in to_find:
        _tot_exp = 0
        _tmp = np.zeros(4)
        for i in range(0, 4):
            val = np.dot(XX[len(XX)-2][i], _set[0:4])
            _tmp[i] = val
            _tot_exp = _tot_exp + val
        _tmp = np.divide(_tmp, _tot_exp)
        _predicted = np.argmax(_tmp)
        #print('Predicted: ', _predicted, ', real: ', _set[4])
        if _predicted != _set[4]:
            wrong_answers = wrong_answers + 1

    print("Wrong predicted values: ", wrong_answers, "/", len(to_find))
    input("Press [enter] to continue.")