import numpy as np
from neural_net_components import back_propagation, forward_pass


def train_once(neural_net, X, Y, lr, momentum, cost_f):

    output = forward_pass(neural_net, X)
    neural_net = back_propagation(neural_net, Y, lr, momentum, cost_f, output)
    return neural_net


def training(epochs, neural_net, X, Y, lr, momentum, fa):
    x1 = X[0]
    y1 = Y[0]

    x1 = np.atleast_2d(x1)
    y1 = np.atleast_2d(y1)

    # Entrenamos a la red!
    trained_neural_net = train_once(
        neural_net, x1, y1, lr, momentum, fa.costo_derivada)
    for i in range(epochs):
        for x, y in zip(X, Y):
            x = np.atleast_2d(x)
            y = np.atleast_2d(y)

            trained_neural_net = train_once(
                trained_neural_net, x, y, lr, momentum, fa.costo_derivada)
    # print(trained_neural_net[3].pesos)
    return trained_neural_net


def predict(trained_neural_net, patron):
    # print(trained_neural_net)
    result = forward_pass(trained_neural_net, patron)
    return result[-1][1]
