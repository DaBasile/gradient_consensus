from mpi4py import MPI
import create_matrix as cm
import numpy as np
import matplotlib.pyplot as plt
import sys
import functions as func
import time
import networkx as nx

usage_message = "usage mpiexec -n (number of agent) python3 filename.py -f \"function name\"" \
                 " optional[" \
                 "-k (number of connection per agent) " \
                 "-a (alpha type: \"diminishing\"/\"constant\") (alpha exp coefficient) (psi coefficient) " \
                 "-e (epsilon)" \
                 "-n (normalized [only for softmax])" \
                 "-m (file path to matrix WARNING possibly not doubly stochastic)]"

################################################
### Check if all data are inserted correctly ###
################################################

function_name = "softmax"
alpha_type = "diminiscing"
alpha_exp_coefficient = 0.01
psi_coefficient = 0.01
epsilon = 0.001
number_of_inn_connection = 1
normalized = False


for i, arg in enumerate(sys.argv):

    if arg == "-f":
        if sys.argv[i + 1] == "softmax" or sys.argv[i + 1] == "quadratic" or sys.argv[i + 1] == "exponential":
            function_name = str(sys.argv[i + 1])

        else:
            sys.exit("admitted function are: \n \"softmax\" \n \"quadratic\" \n \"exponential\"")

        if function_name == "exponential":
            alpha_exp_coefficient = 0.000000000001
            psi_coefficient = 0.000000001

    elif arg == "-k":
        try:
            number_of_inn_connection = int(sys.argv[i + 1])
        except (ValueError, IndexError):
            print("Value error at -k")
            sys.exit(usage_message)

    elif arg == "-a":
        try:
            alpha_type = str(sys.argv[i + 1])
            alpha_exp_coefficient = float(sys.argv[i + 2])
            psi_coefficient = float(sys.argv[i + 3])
        except (ValueError, IndexError):
            print("Value error at -a")
            sys.exit(usage_message)

    elif arg == "-e":
        try:
            epsilon = float(sys.argv[i + 1])
        except (ValueError, IndexError):
            print("Value error at -e")
            sys.exit(usage_message)

    elif arg == "-n":
        normalized = True

#########################
### START MPI PROGRAM ###
#########################

# --oversubscribe -n 6
dataset = np.loadtxt('iris_training_complete.txt', delimiter=';', dtype=float)
#matrix = np.loadtxt('matrix.txt', delimiter=';', dtype=float)

""" Define world parameter, these have been got from mpi system """
world = MPI.COMM_WORLD
agents_number = world.Get_size()
rank = world.Get_rank()

""" Define variables """
MAX_ITERATIONS = 10000
category_n = 3
dimensions = [category_n, 4]

# Assign dataset to each agent
dataset_portion = len(dataset) / agents_number
start_dataset = rank * dataset_portion
end_dataset = (rank * dataset_portion) + dataset_portion
start_dataset = int(start_dataset)
end_dataset = int(end_dataset)
personal_dataset = dataset[start_dataset:end_dataset]

print("Agent ", rank, " got ", len(personal_dataset), " rows of dataset")

world.Barrier()
sys.stdout.flush()

graph = cm.createAdjM(agents_number, number_of_inn_connection)

if rank == 0:
    plt.figure('Graph')
    nx.draw(graph, with_labels=True)
    plt.draw()
    plt.show()

x0 = np.ones(dimensions)

XX = np.zeros([MAX_ITERATIONS, *dimensions])
losses = np.zeros(MAX_ITERATIONS)
XX[0] = x0
losses[0] = func.loss_softmax(x0, category_n, personal_dataset)

num_of_neighbors = 0
for in_neighbors in graph.predecessors(rank):
    num_of_neighbors = num_of_neighbors + 1
weight = 1 / (num_of_neighbors + 1)  # 1 is for self-loop

world.Barrier()

if function_name == "quadratic":
    Q = cm.createQ(dimensions[1])
    r = cm.createR(dimensions[1])

epsilon_reached = False
buff = False

ITERATION_DONE = 0

start_time = time.time()

for tt in range(1, MAX_ITERATIONS - 1):

    if alpha_type == "diminishing":
        alpha = psi_coefficient * (1 / tt) ** alpha_exp_coefficient
    else:
        alpha = alpha_exp_coefficient

    # Update with my previous state
    u_i = np.multiply(XX[tt - 1], weight)

    # for i, j in matrix:
    #     if matrix[i][j] != 0:
    #         world.send(XX[tt - 1], dest=j)
    #
    # for i, j in matrix:
    #     print("i = ", i, " j = ", j)
    #     if matrix[j][i] != 0:
    #         u_i = u_i + world.recv(source=i) * matrix[j][i]

    # Send the state to neighbors
    for node in graph.successors(rank):
        world.send(XX[tt - 1], dest=node)

    # Update with state of all nodes before me
    for node in graph.predecessors(rank):
        u_i = u_i + world.recv(source=node) * weight

    # Go in the opposite direction with respect to the gradient
    gradient = 0
    # print(XX[tt-1])

    if function_name == "softmax":
        gradient = func.gradient_softmax(XX[tt - 1], category_n, dimensions, personal_dataset,
                                         normalized)

    elif function_name == "quadratic":
        gradient = func.gradient_quadratic(XX[tt - 1], category_n, dimensions, personal_dataset, Q, r)

    elif function_name == "exponential":
        gradient = func.gradient_exponential(XX[tt - 1], category_n, dimensions, personal_dataset)

    grad = np.multiply(alpha, gradient)

    for i in range(0, dimensions[0]):
        u_i[i] = np.subtract(u_i[i], grad[i])

    # Store  my new state
    XX[tt] = u_i

    if function_name == "softmax":
        losses[tt] = func.loss_softmax(XX[tt], category_n, personal_dataset)

    elif function_name == "quadratic":
        losses[tt] = func.loss_quadratic(XX[tt], category_n, dimensions, personal_dataset, Q, r)

    elif function_name == "exponential":
        losses[tt] = func.loss_exponential(XX[tt - 1], category_n, dimensions, personal_dataset)

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
        if rank == 0:
            print("Iteration ", tt, "/", MAX_ITERATIONS)
            sys.stdout.flush()
    ITERATION_DONE = tt

exec_time = time.time() - start_time

# synchronise
world.Barrier()

# print("Parameters of node ", rank)
# print(XX[ITERATION_DONE - 3])

sys.stdout.flush()

if rank != 0:
    world.send(losses, dest=0)
    world.send(XX, dest=0)


########################################
### Starting centralized calculation ###
########################################
if rank == 0:
    optimal_value = 8.9450

    # Take the losses from all the other agents and sum
    # We now have the overall loss given from the cost function
    for i in range(1, agents_number):
        agent_loss = world.recv(source=i)
        losses = np.add(losses, agent_loss)

    # Print with respect to 0
    for i in range(0, len(losses)):
        losses[i] = np.subtract(losses[i], optimal_value)

    XX_agents = np.zeros([agents_number, *[MAX_ITERATIONS, *dimensions]])
    XX_agents[0] = XX
    for i in range(1, agents_number):
        XX_agents[i] = world.recv(source=i)

    # Plot cost function
    plt.figure()
    plt.plot(range(0, ITERATION_DONE - 3), losses[0:ITERATION_DONE - 3])
    plt.axhline(y=0, color="blue", linestyle="dashed")
    plt.yscale('log')
    plt.grid(True)
    plt.title("$\log{ \sum_{i=0}^" + str(agents_number) + " f_i }$")
    plt.show()

    # Plot cost function
    plt.figure()
    plt.plot(range(0, ITERATION_DONE - 3), losses[0:ITERATION_DONE - 3])
    plt.axhline(y=optimal_value, color="blue", linestyle="dashed")
    plt.grid(True)
    plt.title("$\sum_{i=0}^" + str(agents_number) + " f_i$")
    plt.show()

    to_find = np.loadtxt('iris_training.txt', delimiter=';', dtype=float)

    wrong_answers = 0
    for _set in to_find:
        _tot_exp = 0
        _tmp = np.zeros(4)
        for i in range(0, category_n):
            const_to_subtract = func.find_const_to_subtract(XX[ITERATION_DONE-2], _set[0:4])
            val = np.exp(np.dot(XX[ITERATION_DONE - 2][i], _set[0:4]) - const_to_subtract)
            _tmp[i] = val
            _tot_exp = _tot_exp + val
        _tmp = np.divide(_tmp, _tot_exp)
        _predicted = np.argmax(_tmp)
        # print('Predicted: ', _predicted, ', real: ', _set[4])
        if _predicted != _set[4]:
            wrong_answers = wrong_answers + 1

    # Show consensus
    for category in range(0, category_n):
        index = 1
        for component in range(0, 4):
            figure = plt.figure()
            plt.title("$\Theta_{" + str(category) + "}$, component " + str(index))
            index = index + 1
            for agent in range(0, agents_number):
                if agent == 0:
                    color = "blue"
                elif agent == 1:
                    color = "green"
                elif agent == 2:
                    color = "yellow"
                else:
                    color = "red"
                plt.plot(range(0, ITERATION_DONE), XX_agents[agent][0:ITERATION_DONE, 0, component], color=color)
            plt.show()

    print("Iteration done: ", ITERATION_DONE, " Agent number: ", agents_number, "\nEpsilon: ", epsilon,
          " Const  Alpha: ", alpha_exp_coefficient, " Const Psi ", psi_coefficient,
          "\nExecution time: ", exec_time, " Wrong preditions: ", wrong_answers)

    input("Press [enter] to continue.")

