import numpy as np
# import funciones_act as fa

# defino clase para una capa
class capa:
    # inicio de la clase
    def __init__(self, n_conexiones, n_neuronas, funcion_act, funcion_act_der):

        # inicializo funciones de activacion

        self.act_f = funcion_act
        self.act_f_derivada = funcion_act_der
        # inicializo pesos y bias
        self.pesos = np.random.rand(n_conexiones, n_neuronas)*1-1
        self.bias = np.random.rand(1, n_neuronas)*1-1
        # matriz para los pesos de la capa anterior
        self.pesos_anterior = np.zeros((n_conexiones, n_neuronas))

# defino una funcion para crear la red
def crear_red(topologia, fa):
    # inicializo una lista para guardar cada capa
    red = []
    # recorro cada una de las capas
    for l, layer in enumerate(topologia[:-1]):
        # agrego la capas ocultas a la red
        if (l < 2):
            red.append(capa(topologia[l], topologia[l+1], fa.lineal, fa.lineal_derivada))
        # agrego la capa de salida a la red
        else:
            red.append(capa(topologia[l], topologia[l+1], fa.sigmoide, fa.sigmoide_derivada))
    # retorno la red
    return red

def forward_pass(neural_net, X):
    # inicializo una lista para guardar las salidas de cada capa
    output = [(None, X)]

    # recorro cada una de las capas
    # foorward pass
    for l, layer in enumerate(neural_net):
        # print(l)
        # obtengo la entrada de la capa
        pre_activacion = output[-1][1] @ neural_net[l].pesos + neural_net[l].bias
        # obtengo la salida de la capa
        post_actiavacion = neural_net[l].act_f(pre_activacion)
        # agrego la salida a la lista de output
        output.append((pre_activacion, post_actiavacion))
    return output

def back_propagation(neural_net, Y, lr, momentum, cost_f, outputs):
    # backpropagation
    # inicializo una lista para guardar los errores
    errores = []
    # recorro cada una de las capas en orden inverso
    for l in reversed(range(0, len(neural_net))):
        # obtengo  las salidas de de la ultima capa
        pre_activ = outputs[l+1][0]
        post_activ = outputs[l+1][1]
        # trato la ultima capa
        if l == len(neural_net)-1:
            # calculo el error
            errores.insert(0, cost_f(post_activ, Y)
                            * neural_net[l].act_f_derivada(post_activ))
        # trato las capas ocultas
        else:
            # calculo el error
            errores.insert(0, errores[0] @ W.T *
                            neural_net[l].act_f_derivada(post_activ))
        W = neural_net[l].pesos
        # descenso del gradiente
        neural_net[l].bias = neural_net[l].bias - \
            np.mean(errores[0], axis=0, keepdims=True) * \
            lr

        # guardo pesos anteriores
        pesos_ateriores = (outputs[l][1].T @ errores[0] * lr) + (
            momentum * neural_net[l].pesos_anterior)
        # actualizo pesos
        neural_net[l].pesos = neural_net[l].pesos - pesos_ateriores
        # actualizo pesos anteriores
        neural_net[l].pesos_anterior = pesos_ateriores

        # Esto dejo comentado por las dudas
        # neural_net[l].pesos = neural_net[l].pesos - \
        #     outputs[l][1].T @ errores[0] * lr
    return neural_net