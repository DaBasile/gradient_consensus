import numpy as np


def quadraticF(x, Q, r):
    x_t = x
    r_t = np.zeros(1, len(x))
    for i in range(0, len(x)):
        x_t[1][i] = x[i]

    for i in range(0, len(r)):
        r_t[1][i] = r[i]


    f = 1/ 2 * np.dot(np.dot(x_t, Q), x) + np.dot(r_t, x)

    return f


def gradientF(x, Q, r):
    for i in range(0, len(Q)):
        for j in range(0, len(Q)):
            Q_t[i][j] = Q[j][i]

    df = 1 / 2 * np.dot(Q_t, x) + 1 / 2 * np.dot(Q, x) + r

    return df


def newPositionX(U_i, X, Q, r, alpha):
    return U_id - alfa * gradientF(X, Q, r);
