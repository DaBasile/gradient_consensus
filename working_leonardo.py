from mpi4py import MPI
import create_matrix as cm
import numpy as np
import matplotlib.pyplot as plt
import sys


CONSTANT_TO_SUBTRACT = 15


def loss(all_theta, category_count):
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
            thetas[category] = thetas[category] - 1 / len(personal_dataset) * np.multiply(personal_dataset[index][:4], coeff)
    return thetas


# def augument_feature_vector(_dataset):
#     _set = np.zeros([len(_dataset), 6])
#     for index in range(0, len(_dataset) - 1):
#         _set[index] = [1, *_dataset[index]]
#     return _set

# --oversubscribe -n 6
dataset = np.loadtxt('iris_training_complete.txt', delimiter=';', dtype=float)
# dataset = normalize_dataset(dataset)

""" Define world parameter, these have been got from mpi system """
world = MPI.COMM_WORLD
agents_number = world.Get_size()
rank = world.Get_rank()

""" Define variables """
MAX_ITERATIONS = 10000
dimensions = [3, 4]
# dimensions = [3, 5]
epsilon = 0.00001
category_n = 3

# Assign dataset to each agent
dataset_portion = len(dataset) / agents_number
start_dataset = int(rank * dataset_portion)
end_dataset = int ((rank * dataset_portion) + dataset_portion)
personal_dataset = dataset[start_dataset:end_dataset]


print("Agent ", rank, " got ", len(personal_dataset), " rows of dataset")

world.Barrier()
sys.stdout.flush()

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

# epsilon_reached = [False for i in range(agents_number)]
epsilon_reached = False
buff = False

ITERATION_DONE = 0


for tt in range(1, MAX_ITERATIONS - 1):

    alpha = 0.01 * (1 / tt) ** 0.5

    # Update with my previous state
    u_i = np.multiply(XX[tt - 1], weight)

    # Send the state to neighbors
    for node in adj.successors(rank):
        world.send(XX[tt - 1], dest=node)

    # Update with state of all nodes before me
    for node in adj.predecessors(rank):
        u_i = u_i + world.recv(source=node) * weight

    # Go in the opposite direction with respect to the gradient
    grad = np.multiply(alpha, gradientF(XX[tt - 1], category_n))
    for i in range(0, dimensions[0]):
        u_i[i] = np.subtract(u_i[i], grad[i])
    # Store  my new state
    XX[tt] = u_i

    losses[tt] = loss(XX[tt], category_n)

    # Checking epsilon reached condition
    if np.linalg.norm(np.subtract(XX[tt], XX[tt - 1])) < epsilon:
        buff = True

    # Rank 0 get all epsilon and check if all reached it
    buffer = world.gather(buff, root=0)

    # If true it set epsilon reached
    if rank == 0:
        if False not in buffer:
            epsilon_reached = True

    # Send epsilon reached to all agents
    epsilon_reached = world.bcast(epsilon_reached, root=0)

    # Check if all agent have reached epsilon condition and then exit from loop
    if epsilon_reached:
        if rank == 0:
            print("Exiting at iteration ", tt, "/", MAX_ITERATIONS, "Condition on epsilon reached")
            sys.stdout.flush()
        
        break

    if tt in range(0, MAX_ITERATIONS, 100):
        if rank == 0 :
            print("Iteration ", tt, "/" , MAX_ITERATIONS)
            sys.stdout.flush()
    ITERATION_DONE = tt

# synchronise
world.Barrier()

print("Parameters of node ", rank)
print(XX[ITERATION_DONE - 3])

world.Barrier()
sys.stdout.flush()


if rank != 0:
    world.send(losses, dest=0)
    world.send(XX, dest=0)

if rank == 0:

    # Take the losses from all the other agents and sum
    # We now have the overall loss given from the cost function
    for i in range(1, agents_number):
        agent_loss = world.recv(source=i)
        losses = np.add(losses, agent_loss)

    XX_agents = np.zeros([agents_number, *[MAX_ITERATIONS, *dimensions]])
    XX_agents[0] = XX
    for i in range(1, agents_number):
        XX_agents[i] = world.recv(source=i)

    log_losses = np.zeros(len(losses))
    for index in range(0, len(losses)):
        log_losses[index] = np.log(losses[index] - 8.945)

    # Plot cost function logarithmic
    plt.figure()
    plt.plot(range(0, ITERATION_DONE - 3), log_losses[0:ITERATION_DONE - 3])
    plt.title("$\sum_{i=0}^" + str(agents_number) + " f_i$")
    plt.show()

    # Plot cost function
    plt.figure()
    plt.plot(range(0, ITERATION_DONE - 3), losses[0:ITERATION_DONE - 3])
    plt.title("$\sum_{i=0}^" + str(agents_number) + " f_i$")
    plt.show()
    #plt.pause(100)

if rank == 0:
    to_find = np.loadtxt('iris_training.txt', delimiter=';', dtype=float)
    # to_find = normalize_dataset(to_find)
    wrong_answers = 0
    for _set in to_find:
        _tot_exp = 0
        _tmp = np.zeros(4)
        for i in range(0, category_n):
            val = np.exp(np.dot(XX[ITERATION_DONE - 2][i], _set[0:4]) - CONSTANT_TO_SUBTRACT)
            _tmp[i] = val
            _tot_exp = _tot_exp + val
        _tmp = np.divide(_tmp, _tot_exp)
        _predicted = np.argmax(_tmp)
        # print('Predicted: ', _predicted, ', real: ', _set[4])
        if _predicted != _set[4]:
            wrong_answers = wrong_answers + 1

    print("Wrong predicted values: ", wrong_answers, "/", len(to_find))

    # Show consensus
    #for category in range(0, category_n):
    for component in range(0, 4):
        figure = plt.figure()
        for agent in range(0, agents_number):
            label = "Agent " + str(agent)
            plt.plot(range(0, ITERATION_DONE), XX_agents[agent][0:ITERATION_DONE, 0, component], label=label)
        plt.title("Component #"+str(component)+" of each $\Theta_{0}^{i}$")
        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.show()

    plt.pause(200000)

    # XX_theta_0 = XX[:, 0] - XX[:, 2]
    # XX_theta_1 = XX[:, 1] - XX[:, 2]
    # XX_theta_2 = XX[:, 2] - XX[:, 2]

    # plt.plot(range(0, ITERATION_DONE), XX_theta_0[0:ITERATION_DONE, 0], label="calculated")
    # plt.axhline(y=1.0229*(10**3), label='$\Theta_{0}^{0}$')
    # plt.plot(range(0, ITERATION_DONE), XX_theta_0[0:ITERATION_DONE, 1], label="calculated")
    # plt.axhline(y=0.8349*(10**3), label='$\Theta_{0}^{1}$')
    # plt.plot(range(0, ITERATION_DONE), XX_theta_0[0:ITERATION_DONE, 2], label="calculated")
    # plt.axhline(y=2.2392*(10**3), label='$\Theta_{0}^{2}$')
    # plt.plot(range(0, ITERATION_DONE), XX_theta_0[0:ITERATION_DONE, 3], label="calculated")
    # plt.axhline(y=-1.3623*(10**3), label='$\Theta_{0}^{3}$')
    # plt.plot(range(0, ITERATION_DONE), XX_theta_0[0:ITERATION_DONE, 4], label="calculated")
    # plt.axhline(y=-6.1496*(10**3), label='$\Theta_{0}^{4}$')
    # #leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    # #leg.get_frame().set_alpha(0.5)
    # # plt.plot(range(0, ITERATION_DONE), 1.0229, label="expected")
    # plt.title("Theta 0")
    # plt.show()

    # Print of the values

    input("Press [enter] to continue.")
