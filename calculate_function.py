import numpy as np


def quadraticF(x, Q, r):

    f = 1 / 2 * np.dot(np.dot(np.transpose(x), Q), x) + np.dot(np.transpose(r), x)

    return f


def gradientF(x, Q, r):

    df = 1 / 2 * np.dot(np.transpose(Q), x) + 1 / 2 * np.dot(Q, x) + r

    return df


def newPositionX(xk, X, Q, r, alpha):

    xk_1 = xk - alpha * gradientF(X, Q, r)

    return xk_1

