from mpi4py import MPI
import create_matrix as cm
import numpy as np
import matplotlib.pyplot as plt
import sys
import functions as func
import time

CONSTANT_TO_SUBTRACT = 15

usage_message = "usage mpiexec -n (number of agent) python3 filename.py -f \"function name\"" \
                 " optional[-k (number of connection per agent) " \
                 "-a (alpha type \"diminishing\"/\"constant\") (alpha coefficient) (psi coefficient) " \
                 "-e (epsilon)]"

################################################
### Check if all data are inserted correctly ###
################################################
if len(sys.argv) <= 1:
    sys.exit(usage_message)

if sys.argv[1] == "-f":
    if sys.argv[2] == "softmax" or sys.argv[2] == "quadratic" or sys.argv[2] == "exponential":
        function_name = str(sys.argv[2])

    else:
        sys.exit("admitted function are: \n \"softmax\" \n \"quadratic\" \n")

    alpha_type = "diminiscing"
    alpha_exp_coefficient = 0.01
    psi_coefficient = 0.01
    epsilon = 0.01
    number_of_inn_connection = 1

else:
    sys.exit(usage_message)

if len(sys.argv) >= 4:
    if sys.argv[3] == "-k":
        number_of_inn_connection = int(sys.argv[4])

        if len(sys.argv) >= 6:
            if sys.argv[5] == "-a":
                alpha_type = sys.argv[6]
                alpha_exp_coefficient = float(sys.argv[7])
                psi_coefficient = float(sys.argv[8])

                if len(sys.argv) >= 10:
                    if sys.argv[9] == "-e":
                        epsilon = float(sys.argv[10])

                    else:
                        sys.exit(usage_message)

            elif sys.argv[5] == "-e":
                epsilon = float(sys.argv[6])

                if len(sys.argv) >= 8:
                    if sys.argv[7] == "-a":
                        alpha_type = sys.argv[8]
                        alpha_exp_coefficient = float(sys.argv[9])
                        psi_coefficient = float(sys.argv[10])

                    else:
                        sys.exit(usage_message)

            else:
                sys.exit(usage_message)

    elif sys.argv[3] == "-a":
        alpha_type = sys.argv[4]
        alpha_exp_coefficient = float(sys.argv[5])
        psi_coefficient = float(sys.argv[6])

        if len(sys.argv) >= 8:
            if sys.argv[7] == "-k":
                number_of_inn_connection = int(sys.argv[8])

                if len(sys.argv) >= 10 and sys.argv[9] == "-e":
                    epsilon = float(sys.argv[10])

                else:
                    sys.exit(usage_message)

            elif sys.argv[7] == "-e":
                epsilon = float(sys.argv[8])

                if len(sys.argv) >= 10:
                    if sys.argv[9] == "-k":
                        number_of_inn_connection = int(sys.argv[10])

                    else:
                        sys.exit(usage_message)

            else:
                sys.exit(usage_message)

    elif sys.argv[3] == "-e":
        epsilon = float(sys.argv[4])

        if len(sys.argv) >= 6:
            if sys.argv[5] == "-k":
                number_of_inn_connection = int(sys.argv[6])

                if len(sys.argv) >= 8 and sys.argv[7] == "-a":
                    alpha_type = sys.argv[8]
                    alpha_exp_coefficient = float(sys.argv[9])
                    psi_coefficient = float(sys.argv[10])

                else:
                    sys.exit(usage_message)

            elif sys.argv[5] == "-a":
                alpha_type = sys.argv[6]
                alpha_exp_coefficient = float(sys.argv[7])
                psi_coefficient = float(sys.argv[8])

                if len(sys.argv) >= 10:
                    if sys.argv[9] == "-k":
                        number_of_inn_connection = int(sys.argv[10])

                    else:
                        sys.exit(usage_message)

            else:
                sys.exit(usage_message)

    else:
        sys.exit(usage_message)

#########################
### START MPI PROGRAM ###
#########################

# --oversubscribe -n 6
dataset = np.loadtxt('iris_training_complete.txt', delimiter=';', dtype=float)

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

adj = cm.createAdjM(agents_number, number_of_inn_connection)

x0 = np.ones(dimensions)

XX = np.zeros([MAX_ITERATIONS, *dimensions])
losses = np.zeros(MAX_ITERATIONS)
XX[0] = x0

num_of_neighbors = 0
for in_neighbors in adj.predecessors(rank):
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

    # Send the state to neighbors
    for node in adj.successors(rank):
        world.send(XX[tt - 1], dest=node)

    # Update with state of all nodes before me
    for node in adj.predecessors(rank):
        u_i = u_i + world.recv(source=node) * weight

    # Go in the opposite direction with respect to the gradient
    gradient = 0

    if function_name == "softmax":
        gradient = func.gradient_softmax(XX[tt - 1], category_n, dimensions, personal_dataset, CONSTANT_TO_SUBTRACT)

    elif function_name == "quadratic":
        gradient = func.gradient_quadratic(XX[tt - 1], category_n, dimensions, personal_dataset, Q, r)

    elif function_name == "exponential":
        gradient = func.gradient_exponential(XX[tt - 1], category_n, dimensions, personal_dataset, CONSTANT_TO_SUBTRACT)

     #print(gradient)

    grad = np.multiply(alpha, gradient)

    for i in range(0, dimensions[0]):
        u_i[i] = np.subtract(u_i[i], grad[i])

    # Store  my new state
    XX[tt] = u_i

    if function_name == "softmax":
        losses[tt] = func.loss_softmax(XX[tt], category_n, personal_dataset, CONSTANT_TO_SUBTRACT)

    elif function_name == "quadratic":
        losses[tt] = func.loss_quadratic(XX[tt], category_n, dimensions, personal_dataset, Q, r)

    elif function_name == "exponential":
        losses[tt] = func.loss_exponential(XX[tt - 1], category_n, dimensions, personal_dataset, CONSTANT_TO_SUBTRACT)

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

# synchronise
world.Barrier()

print("Parameters of node ", rank)
print(XX[ITERATION_DONE - 3])

world.Barrier()
sys.stdout.flush()

if rank != 0:
    world.send(losses, dest=0)
    world.send(XX, dest=0)


########################################
### Starting centralized calculation ###
########################################
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
    for index in range(0, len(losses)-1):
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
    # for category in range(0, category_n):
    for component in range(0, 4):
        figure = plt.figure()
        for agent in range(0, agents_number):
            label = "Agent " + str(agent)
            plt.plot(range(0, ITERATION_DONE), XX_agents[agent][0:ITERATION_DONE, 0, component], label=label)
        plt.title("Component #" + str(component) + " of each $\Theta_{0}^{i}$")
        leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.show()
    print("Iteration done: ", ITERATION_DONE, " Agent number: ", agents_number, "\nEpsilon: ", epsilon,
          " Const  Alpha: ", alpha_exp_coefficient, " Const Psi ", psi_coefficient,
          "\nExecution time: ", time.time() - start_time, " Wrong preditions: ", wrong_answers)

    plt.pause(20)
    input("Press [enter] to continue.")

