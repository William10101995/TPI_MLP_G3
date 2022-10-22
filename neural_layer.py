from copyreg import constructor
import numpy as np
from dataset_gen import create_dataset, pattern_F, pattern_B, pattern_D
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from sklearn.datasets import make_circles


dataset = create_dataset(1500, [pattern_B, pattern_D, pattern_F])

input_X = np.array(dataset[0])
input_Y = np.array(dataset[1])[:, np.newaxis]

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


topology = [p, 5, 3]

neural_net = create_nn(topology, sigm)

l2_cost = (
    lambda Yp, Yr: np.mean((Yp - Yr) ** 2), #Yp: Y predicha, Yr: Y real
    lambda Yp, Yr: (Yp - Yr)
)


def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
  
  
    out = [(None, X)]

    # Forward pass
    for l, layer in enumerate(neural_net):
        
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b # Producto matricial con el @
        a = neural_net[l].act_f[0](z)
        out.append((z, a))
    #print(l2_cost[0](out[-1][1],  Y))

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


neural_n = create_nn(topology, sigm)

loss = [] 
for i in range(4000):
  # Entrenamos a la red!

  pY = train(neural_n, input_X, input_Y, l2_cost, lr=0.01)

  if i % 25 == 0: # Cada 25 iteraciones, ejecuta lo de adentro
    loss.append(l2_cost[0](pY, input_Y))
    clear_output(wait=True)
    plt.plot(range(len(loss)), loss)
    plt.show()
    time.sleep(0.5)

print(loss)
