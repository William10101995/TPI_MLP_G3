from copyreg import constructor
import numpy as np
from dataset_gen import create_dataset, pattern_F, pattern_B, pattern_D
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from sklearn.datasets import make_circles


dataset = create_dataset(1000, [pattern_B, pattern_D, pattern_F])

input_X = np.array(dataset[0])
input_Y = np.array(dataset[1])[:, np.newaxis]

# plt.scatter(input_X[input_Y[:, 0] == 0, 0], input_X[input_Y[:, 0] == 0, 1], c="skyblue")
# plt.scatter(input_X[input_Y[:, 0] == 1, 0], input_X[input_Y[:, 0] == 1, 1], c="salmon")
# plt.axis("equal")
# plt.show()


p = 100
# CLASE DE LA CAPA DE LA RED

class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur) * 2 - 1          
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1     


# FUNCIONES DE ACTIVACION

sigm = (
    lambda x: 1 / (1 + np.e ** (-x)),   # funcion de activacion
    lambda x: x * (1 - x)               # derivada
)


def create_nn(topology, act_f):
  
    nn = []

    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l + 1], act_f))
    return nn


topology = [p, 4, 8, 3]

neural_net = create_nn(topology, sigm)

l2_cost = (
    lambda Yp, Yr: np.mean((Yp - Yr) ** 2), #Yp: Y predicha, Yr: Y real
    lambda Yp, Yr: (Yp - Yr)
)


def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
  
  
    out = [(None, X)]

    # Forward pass
    for l, layer in enumerate(neural_net):
        print("out  ---> " + str(out[-1][1].shape))
        # print(out[-1][1])
        

        # print("a  ---> " + str(neural_net[l].act_f[0](out[-1][1] @ neural_net[l].W + neural_net[l].b).shape))
        # print(neural_net[l].act_f[0](out[-1][1] @ neural_net[l].W + neural_net[l].b))
        print("W    ---> " + str(neural_net[l].W.shape))
        # print(neural_net[l].W)

        # print("b    ---> " + str(neural_net[l].b.shape))
        # print(neural_net[l].b)
        # print()
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b # Producto matricial con el @
        a = neural_net[l].act_f[0](z)
        # print(a)
        out.append((z, a))
    # print(l2_cost[0](out[-1][1],  Y))

    if train:
        # Backward pass
        deltas = []

        for l in reversed(range(0, len(neural_net))):

            z = out[l + 1][0]
            a = out[l + 1][1]

            if l == len(neural_net) - 1: # Ultima capa
                # Calcular delta ultima capa
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
            else:
                # Calcular delta respecto a capa previa
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].W

            # Gradient descent
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr

    return out[-1][1]

# train(neural_net, input_X, input_Y, l2_cost, 0.5)





neural_n = create_nn(topology, sigm)

loss = [] 
for i in range(1000):
  # Entrenamos a la red!

  pY = train(neural_n, input_X, input_Y, l2_cost, lr=0.05)

  if i % 25 == 0: # Cada 25 iteraciones, ejecuta lo de adentro
    loss.append(l2_cost[0](pY, input_Y))

    res = 50

    _x0 = np.linspace(-5, 5, res)
    _x1 = np.linspace(-5, 5, res)

    _Y = np.zeros((res, res))

    for i0, x0 in enumerate(_x0):
      for i1, x1 in enumerate(_x1):
        _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), input_Y, l2_cost, train=False)[0][0]
    
    plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
    plt.axis("equal")

    plt.scatter(input_X[input_Y[:, 0] == 0, 0], input_X[input_Y[:, 0] == 0, 1], c="skyblue")
    plt.scatter(input_X[input_Y[:, 0] == 1, 0], input_X[input_Y[:, 0] == 1, 1], c="salmon")

    # clear_output(wait=True)
    plt.show()
    # plt.plot(range(len(loss)), loss)
    # plt.show()
    plt.close(1)
    time.sleep(0.5)