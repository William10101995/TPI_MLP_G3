import numpy as np
# import funciones_act as fa

# defino clase para una capa
class layer:
    # inicio de la clase
    def __init__(self, n_conexiones, n_neuronas, funcion_act, funcion_act_der):

        # inicializo funciones de activacion
        self.act_f = funcion_act
        self.act_f_derivative = funcion_act_der
        # inicializo W y bias
        self.W = np.random.rand(n_conexiones, n_neuronas)*1-1
        self.bias = np.random.rand(1, n_neuronas)*1-1
        # matriz para los W de la capa anterior
        self.previous_W = np.zeros((n_conexiones, n_neuronas))

# defino una funcion para crear la red
def create_neural_net(topology, fa):
    # inicializo una lista para guardar cada capa
    neural_net = []
    # recorro cada una de las capas
    for l, _layer in enumerate(topology[:-1]):
        # agrego la capas ocultas a la neural_net
        if (l < 2):
            neural_net.append(layer(topology[l], topology[l+1], fa.lineal, fa.lineal_derivada))
        # agrego la layer de salida a la neural_net
        else:
            neural_net.append(layer(topology[l], topology[l+1], fa.sigmoide, fa.sigmoide_derivada))
    # retorno la neural_net
    return neural_net

def forward_pass(neural_net, X):
    # inicializo una lista para guardar las salidas de cada capa
    output = [(None, X)]

    # recorro cada una de las capas
    # foorward pass
    for l, layer in enumerate(neural_net):
        # print(l)
        # obtengo la entrada de la capa
        pre_activ = output[-1][1] @ neural_net[l].W + neural_net[l].bias
        # obtengo la salida de la capa
        post_activ = neural_net[l].act_f(pre_activ)
        # agrego la salida a la lista de output
        output.append((pre_activ, post_activ))
    return output

def back_propagation(neural_net, Y, lr, momentum, cost_f, outputs):
    # backpropagation
    # inicializo una lista para guardar los errors
    errors = []
    # recorro cada una de las capas en orden inverso
    for l in reversed(range(0, len(neural_net))):
        # obtengo  las salidas de de la ultima capa
        pre_activ = outputs[l+1][0]
        post_activ = outputs[l+1][1]
        # trato la ultima capa
        if l == len(neural_net)-1:
            # calculo el error
            errors.insert(0, cost_f(post_activ, Y) * neural_net[l].act_f_derivative(post_activ))
        # trato las capas ocultas
        else:
            # calculo el error
            errors.insert(0, errors[0] @ W.T * neural_net[l].act_f_derivative(post_activ))
        W = neural_net[l].W
        # descenso del gradiente
        neural_net[l].bias = neural_net[l].bias - np.mean(errors[0], axis=0, keepdims=True) * lr

        # guardo pesos anteriores
        previous_W = (outputs[l][1].T @ errors[0] * lr) + (momentum * neural_net[l].previous_W)
        # actualizo pesos
        neural_net[l].W = neural_net[l].W - previous_W
        # actualizo pesos anteriores
        neural_net[l].previous_W = previous_W

        # Esto dejo comentado por las dudas
        # neural_net[l].pesos = neural_net[l].pesos - \
        #     outputs[l][1].T @ errors[0] * lr
    return neural_net