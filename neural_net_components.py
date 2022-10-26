import numpy as np
# defino clase para una capa
class layer:
    # inicio de la clase
    def __init__(self, n_connections, n_neurons, act_f, act_f_der):

        # inicializo funciones de activacion
        self.act_f = act_f
        self.act_f_derivative = act_f_der

        # inicializo pesos y bias
        self.W = np.random.rand(n_connections, n_neurons)*1-1
        self.bias = np.random.rand(1, n_neurons)*1-1

        # matriz para los pesos de la capa anterior
        self.previous_W = np.zeros((n_connections, n_neurons))


def create_neural_net(topology, fa):
    # inicializo una lista para guardar cada capa
    net = []
    # recorro cada una de las capas
    for l, _layer in enumerate(topology[:-1]):
        if(l == len((topology)) - 1):
            net.append(layer(topology[l], topology[l+1], fa.sigmoide, fa.sigmoide_derivative))
        net.append(layer(topology[l], topology[l+1], fa.lineal, fa.lineal_derivative))
    return net

def forward_pass(neural_net, X):
    output = [(None, X)]
    # recorro cada una de las capas
    for l, layer in enumerate(neural_net):
        # obtengo la entrada de la capa
        pre_activ = output[-1][1] @ neural_net[l].W + neural_net[l].bias
        # obtengo la salida de la capa
        post_activ = neural_net[l].act_f(pre_activ)
        # agrego la salida a la lista de salidas
        output.append((pre_activ, post_activ))
    return output


def back_propagation(neural_net, Y, lr, momentum, der_cost_f, outputs):
    errors = []
    # recorro cada una de las capas en orden inverso
    for l in reversed(range(0, len(neural_net))):
        # obtengo  las salidas de de la ultima capa
        pre_activ = outputs[l+1][0]
        post_activ = outputs[l+1][1]
        # trato la ultima capa
        if l == len(neural_net)-1:
            # calculo el error
            errors.insert(0, der_cost_f(post_activ, Y) * neural_net[l].act_f_derivative(post_activ))
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
    return neural_net
