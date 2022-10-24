# importo numpy
import numpy as np
# importo funciones de activacion
import funciones_act as fa
from dataset_gen import cargar_datos, distortion_pattern, pattern_F, pattern_B, pattern_D


# cargo el dataset
dataset = cargar_datos()
# separo el dataset
input_X = np.array(dataset[0])
input_Y = np.array(dataset[1])
test = np.array(dataset[2])
val = np.array(dataset[3])

# Algunos patrones para probar
patronB = np.array(pattern_B).ravel()
patronD = np.array(pattern_D).ravel()
patronF = np.array(pattern_F).ravel()
patD = distortion_pattern(pattern_D, 0.9)
pat = np.array(patD)
patdist = pat.ravel()


# defino clase para una capa
class capa:
    # inicio de la clase
    def __init__(self, n_conexiones, n_neuronas, funcion_act):

        # inicializo funciones de activacion
        if funcion_act == 'sigmoide':
            self.act_f = fa.sigmoide
            self.act_f_derivada = fa.sigmoide_derivada
        elif funcion_act == 'lineal':
            self.act_f = fa.lineal
            self.act_f_derivada = fa.lineal_derivada
        # inicializo pesos y bias
        self.pesos = np.random.rand(n_conexiones, n_neuronas)*1-1
        self.bias = np.random.rand(1, n_neuronas)*1-1


# defino una funcion para crear la red
def crear_red(topologia, funcion_act):
    # inicializo una lista para guardar cada capa
    red = []
    # recorro caada una de las capas
    for l, layer in enumerate(topologia[:-1]):
        # agrego una capa a la red
        red.append(capa(topologia[l], topologia[l+1], funcion_act))
    # retorno la red
    return red


# defino una funcion para entrenar la red
def entrenar(red_neuronal, X, Y, coeficiente_entrenamiento, funcion_costo_der=fa.costo_derivada, entrena=True):
    # inicializo una lista para guardar las salidas de cada capa
    salidas = [(None, X)]

    # recorro cada una de las capas
    # foorward pass
    for l, layer in enumerate(red_neuronal):
        # obtengo la entrada de la capa
        pre_activacion = salidas[-1][1] @ red_neuronal[l].pesos + \
            red_neuronal[l].bias
        # obtengo la salida de la capa
        post_actiavacion = red_neuronal[l].act_f(pre_activacion)
        # agrego la salida a la lista de salidas
        salidas.append((pre_activacion, post_actiavacion))
    if entrena:
        # backpropagation
        # inicializo una lista para guardar los errores
        errores = []
        # recorro cada una de las capas en orden inverso
        for l in reversed(range(0, len(red_neuronal))):
            # obtengo  las salidas de de la ultima capa
            pre_activ = salidas[l+1][0]
            post_activ = salidas[l+1][1]
            # trato la ultima capa
            if l == len(red_neuronal)-1:
                # calculo el error
                errores.insert(0, funcion_costo_der(post_activ, Y)
                               * red_neuronal[l].act_f_derivada(post_activ))
            # trato las capas ocultas
            else:
                # calculo el error
                errores.insert(0, errores[0] @ W.T *
                               red_neuronal[l].act_f_derivada(post_activ))
            W = red_neuronal[l].pesos
            # descenso del gradiente
            red_neuronal[l].bias = red_neuronal[l].bias - \
                np.mean(errores[0], axis=0, keepdims=True) * \
                coeficiente_entrenamiento

            red_neuronal[l].pesos = red_neuronal[l].pesos - \
                salidas[l][1].T @ errores[0] * coeficiente_entrenamiento
    # retorno la ultima salida
    return salidas[-1][1]


# defino una topologia para la red
# 1 entradas, 4 neuronas en la capa oculta, 4 neuronas en la capa oculta, 3 salidas
topologia = [100, 5, 5, 3]
red_neuronal = crear_red(topologia, 'sigmoide')
print("Entrenando red neuronal")
for i in range(2000):
    # Entrenamos a la red!
    entrenar(red_neuronal, input_X, input_Y, 0.05, fa.costo_derivada)
print("Red neuronal entrenada")


# defino una funcion para predecir
def predecir(patron):
    resultado = entrenar(red_neuronal, patron, input_Y, 0.05,
                         fa.costo_derivada, entrena=False)
    return np.rint(resultado)


# predecimos
for i in range(len(test)):
    prediccion = predecir(test[i])
    print("Prediciendo ejemplo numero: ", i)
    if (prediccion[0] == np.array([0, 0, 1])).all():
        print("Su letra es D")
    elif (prediccion[0] == np.array([0, 1, 0])).all():
        print("Su letra es F")
    elif (prediccion[0] == np.array([1, 0, 0])).all():
        print("Su letra es B")
    else:
        print("No se pudo identificar la letra")
