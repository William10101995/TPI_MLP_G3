import numpy as np
class layer:
    # inicio de la clase
    def __init__(self, n_connections, n_neurons, act_f, act_f_derivative):
        # inicializo funciones de activacion
        self.act_f = act_f
        self.act_f_derivative = act_f_derivative

        # inicializo pesos y bias
        self.W = np.random.rand(n_connections, n_neurons)*1-1
        self.bias = np.random.rand(1, n_neurons)*1-1

        # matriz para los pesos de la capa anterior
        self.previous = np.zeros((n_connections, n_neurons))


# defino una funcion para crear la red
def create_neural_net(topology, fa):
    # inicializo una lista para guardar cada capa
    net = []
    # recorro caada una de las capas

    for l, _layer in enumerate(topology[:-1]):
        if(l == len((topology)) - 1):
            net.append(layer(topology[l], topology[l+1], fa.sigmoide, fa.sigmoide_derivative))
        # agrego una capa a la red
        net.append(layer(topology[l], topology[l+1], fa.lineal, fa.lineal_derivative))
    return net


def forward_pass(neural_net, input_data):
    output = [(None, input_data)]
    for l, layer in enumerate(neural_net):
        # obtengo la entrada de la capa
        pre_activation = output[-1][1] @ neural_net[l].W + neural_net[l].bias
        # obtengo la salida de la capa
        post_actiavation = neural_net[l].act_f(pre_activation)
        # agrego la salida a la lista de salidas
        output.append((pre_activation, post_actiavation))
    return output

def back_propagation(neural_net, Y, lr, momentum, cost_f, output):
    # inicializo una lista para guardar los errores
    errors = []
    # recorro cada una de las capas en orden inverso
    for l in reversed(range(0, len(neural_net))):
        # obtengo  las salidas de de la ultima capa
        pre_activation = output[l+1][0]
        post_actiavation = output[l+1][1]
        # trato la ultima capa
        if l == len(neural_net)-1:
            # calculo el error
            errors.insert(0, cost_f(post_actiavation, Y) * neural_net[l].act_f_derivative(post_actiavation))
        else: # trato las capas ocultas 
            # calculo el error
            errors.insert(0, errors[0] @ W.T * neural_net[l].act_f_derivative(post_actiavation))
        W = neural_net[l].W

        # descenso del gradiente
        neural_net[l].bias = neural_net[l].bias - np.mean(errors[0], axis=0, keepdims=True) * lr
        # neural_net[l].W = neural_net[l].W - output[l][1].T @ errors[0] * lr

        # guardo pesos anteriores
        previous_W = (output[l][1].T @ errors[0] * lr) + (momentum * neural_net[l].previous)
        # actualizo pesos
        neural_net[l].W = neural_net[l].W - previous_W
        # actualizo pesos anteriores
        neural_net[l].previous = previous_W            
    return neural_net