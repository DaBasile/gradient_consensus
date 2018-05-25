import numpy as np


def loss(all_theta, category_count, personal_dataset, CONSTANT_TO_SUBTRACT):
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


def softmaxRegression(all_theta, category_count, dimensions, personal_dataset, CONSTANT_TO_SUBTRACT):
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


def quadratic(all_theta, category_count, dimensions, personal_dataset, Q, r):
    thetas = np.zeros(dimensions)

    for index in range(0, len(personal_dataset)):

        for category in range(0, category_count):

            if category == personal_dataset[index][4]:
                thetas[category] = np.sum([
                    np.sum([
                        1 / 2 * np.dot(np.transpose(Q),
                                       np.multiply(all_theta[category], personal_dataset[index][:4])),
                        1 / 2 * np.dot(Q, np.multiply(all_theta[category], personal_dataset[index][:4]))],
                        axis=0), np.transpose(r)], axis=0)
    return thetas


def exponential(all_theta, category_count, dimensions, personal_dataset, CONSTANT_TO_SUBTRACT):
    thetas = np.zeros(dimensions)

    for index in range(0, len(personal_dataset)):

        for category in range(0, category_count):

            if category == personal_dataset[index][4]:
                current_vector = np.dot(all_theta[category], personal_dataset[index][:4])
                np.sinh(current_vector)
                thetas[category] = \
                    np.sum([np.cosh(current_vector),
                           #-np.multiply(np.divide(current_vector, np.linalg.norm(current_vector)),
                           np.sinh(current_vector)])

    return thetas

