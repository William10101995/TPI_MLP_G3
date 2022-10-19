import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

# CREAR EL DATASET

n = 500  # numero de registro en nuyestros datos
p = 2  # cuantas caracteristicas tenemos en nuestro registro de datos

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)

Y = Y[:, np.newaxis]

plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
plt.axis("equal")
plt.show()

print(X[0])
print(Y)

# CLASE DE LA CAPA DE LA RED


class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        # esta relacionado con el bias (sesgo)
        self.b = np.random.rand(1, n_neur) * 2 - 1
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1


# FUNCIONES DE ACTIVACION

sigm = (
    lambda x: 1 / (1 + np.e ** (-x)),  # funcion
    lambda x: x * (1 - x)  # derivada
)


def relu(x): return np.maximum(0, x)


_x = np.linspace(-5, 5, 100)
plt.plot(_x, sigm[0](_x))


# l0 = neural_layer(p, 4, sigm)
# l1 = nerual_layer(4, 8, sigm)


def create_nn(topology, act_f):

    nn = []

    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l + 1], act_f))

    return nn


topology = [p, 4, 8, 16, 8, 4, 1]

neural_net = create_nn(topology, sigm)

l2_cost = (
    lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
    lambda Yp, Yr: (Yp - Yr)
)


def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):

    out = [(None, X)]

    # Forward pass
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + \
            neural_net[l].b  # Producto matricial con el @
        a = neural_net[l].act_f[0](z)

        out.append((z, a))
    print(l2_cost[0](out[-1][1], Y))

    if train:
        # Backward pass
        deltas = []

        for l in reversed(range(0, len(neural_net))):

            z = out[l + 1][0]
            a = out[l + 1][1]

            if l == len(neural_net) - 1:
                # Calcular delta ultima capa
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
            else:
                # Calcular delta respecto a capa previa
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].W

            # Gradient descent
            neural_net[l].b = neural_net[l].b - \
                np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr

    return out[-1][1]


train(neural_net, X, Y, l2_cost, 0.5)
