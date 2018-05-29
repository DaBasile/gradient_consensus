import numpy as np


def loss_softmax(all_theta, category_count, personal_dataset, CONSTANT_TO_SUBTRACT):
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


def gradient_softmax(all_theta, category_count, dimensions, personal_dataset, CONSTANT_TO_SUBTRACT, normalized):
    thetas = np.zeros(dimensions)

    if normalized:
        m = len(personal_dataset)
    else:
        m = 1

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
            thetas[category] = thetas[category] - ((1 / m) *
                                                   np.multiply(personal_dataset[index][:4], coeff))

    return thetas


def loss_quadratic(all_theta, category_count, dimensions, personal_dataset, Q, r):
    thetas = np.zeros(dimensions)
    cat_number = np.zeros(category_count)

    for index in range(0, len(personal_dataset)):

        for category in range(0, category_count):

            if category == personal_dataset[index][4]:
                current_set = np.multiply(all_theta[category], personal_dataset[index][:4])
                thetas[category] = thetas[category] + np.sum([np.sum([1 / 2 * np.dot(np.dot(current_set, Q),
                                                                                     np.transpose(current_set))],
                                                                     axis=0),
                                                              np.dot(current_set, r)], axis=0)
                cat_number[category] = cat_number[category] + 1

    _sum = 0

    for category in range(0, category_count):
        _sum = _sum + np.divide(thetas[category], len(personal_dataset)).sum(dtype=np.float64) / 4
    return _sum


def gradient_quadratic(all_theta, category_count, dimensions, personal_dataset, Q, r):
    thetas = np.zeros(dimensions)

    for index in range(0, len(personal_dataset)):

        for category in range(0, category_count):

            if category == personal_dataset[index][4]:
                thetas[category] = np.sum([thetas[category], np.sum([
                    np.sum([
                        1 / 2 * np.dot(np.transpose(Q),
                                       np.multiply(all_theta[category], personal_dataset[index][:4])),
                        1 / 2 * np.dot(Q, np.multiply(all_theta[category], personal_dataset[index][:4]))],
                        axis=0), np.transpose(r)], axis=0)], axis=0)

    for category in range(0, category_count):
        thetas[category] = np.divide(thetas[category], len(personal_dataset))
    return thetas


def loss_exponential(all_theta, category_count, dimensions, personal_dataset, CONSTANT_TO_SUBTRACT):
    thetas = np.zeros(dimensions)

    for index in range(0, len(personal_dataset)):

        for category in range(0, category_count):

            if category == personal_dataset[index][4]:
                coeff = np.exp(
                    np.dot(all_theta[category], personal_dataset[index][:4]) - CONSTANT_TO_SUBTRACT)
                thetas[category] = thetas[category] - np.multiply(personal_dataset[index][:4], coeff)

    _sum = 0

    for category in range(0, category_count):
        _sum = _sum + np.divide(thetas[category], len(personal_dataset)).sum(dtype=np.float64) / 4
    return _sum


def gradient_exponential(all_theta, category_count, dimensions, personal_dataset, CONSTANT_TO_SUBTRACT):
    thetas = np.zeros(dimensions)

    for index in range(0, len(personal_dataset)):

        for category in range(0, category_count):

            if category == personal_dataset[index][4]:
                coeff = np.exp(
                    np.dot(all_theta[category], personal_dataset[index][:4]))
                thetas[category] = thetas[category] - np.multiply(personal_dataset[index][:4], coeff)

    return thetas

