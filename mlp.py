# importo numpy
import numpy as np
# importo funciones de activacion
import funciones_act as fa
from dataset_gen import cargar_datos, distortion_pattern, pattern_F, pattern_B, pattern_D


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
        # matriz para los pesos de la capa anterior
        self.pesos_anterior = np.zeros((n_conexiones, n_neuronas))


# defino una funcion para crear la red
def crear_red(topologia, funcion_act):
    # inicializo una lista para guardar cada capa
    red = []
    # recorro cada una de las capas
    for l, layer in enumerate(topologia[:-1]):
        # agrego la capas ocultas a la red
        if (l < 2):
            red.append(capa(topologia[l], topologia[l+1], funcion_act))
        # agrego la capa de salida a la red
        else:
            red.append(capa(topologia[l], topologia[l+1],
                            'sigmoide'))
    # retorno la red
    return red


# defino una funcion para entrenar la red
def entrenar(red_neuronal, X, Y, coeficiente_entrenamiento, momentum, funcion_costo_der=fa.costo_derivada, entrena=True):
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

            # guardo pesos anteriores
            pesos_ateriores = (salidas[l][1].T @ errores[0] * coeficiente_entrenamiento) + (
                momentum * red_neuronal[l].pesos_anterior)
            # actualizo pesos
            red_neuronal[l].pesos = red_neuronal[l].pesos - pesos_ateriores
            # actualizo pesos anteriores
            red_neuronal[l].pesos_anterior = pesos_ateriores

            # Esto dejo comentado por las dudas
            # red_neuronal[l].pesos = red_neuronal[l].pesos - \
            #     salidas[l][1].T @ errores[0] * coeficiente_entrenamiento
    # retorno la ultima salida
    return salidas[-1][1]


# cargo el dataset
dataset = cargar_datos(1000, 10)
# separo el dataset
input_X = np.array(dataset[0])
input_Y = np.array(dataset[1])
test = np.array(dataset[2])
val = np.array(dataset[3])

# Algunos patrones para probar
patronB = np.array(pattern_B).ravel()
patronD = np.array(pattern_D).ravel()
patronF = np.array(pattern_F).ravel()
patB = distortion_pattern(pattern_B, 0.3)
pat = np.array(patB)
patdist = pat.ravel()

# defino una topologia para la red
# 100 entradas, 5 neuronas en la capa oculta, 5 neuronas en la capa oculta, 3 salidas
topologia = [100, 10, 10, 3]
red_neuronal = crear_red(topologia, 'lineal')

print("Entrenando red neuronal")
for i in range(1000):
    for x, y in zip(input_X, input_Y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        # Entrenamos a la red!
        entrenar(red_neuronal, x, y, 0.5, 0.5, fa.costo_derivada)
print("Red neuronal entrenada")


# defino una funcion para predecir
def predecir(patron):
    resultado = entrenar(red_neuronal, patron, input_Y, 0.5, 0.5,
                         fa.costo_derivada, entrena=False)
    return resultado


# Funciones de prueba
# Funcion para predecir patrones originales y un distorsionado
def predecirPatronesOriginales(patB, patD, patF, patDist):
    print("Prediciendo patrones originales")
    print("Patron B: ", predecir(patB)[0])
    print("Patron D: ", predecir(patD)[0])
    print("Patron F: ", predecir(patF)[0])
    print("Prediccion B Distorsionado 30%: ", predecir(patDist)[0])


# Ejecucion
#predecirPatronesOriginales(patronB, patronD, patronF, patdist)

# Funcion para predecir set de test


def predecirSetTest(test):
    print("Prediciendo set de test")
    for i in range(len(test)):
        print("Patron ", i, ": ", predecir(test[i])[0])

# Ejecucion
# predecirSetTest(test)

# Funcion para predecir set de validacion


def predecirSetValidacion(val):
    print("Prediciendo set de validacion")
    for i in range(len(val)):
        print("Patron ", i, ": ", predecir(val[i])[0])

# Ejecucion
# predecirSetValidacion(val)
