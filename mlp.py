# importo numpy
import numpy as np
# importo funciones de activacion
import funciones_act as fa
from dataset_gen import cargar_datos, distortion_pattern, pattern_F, pattern_B, pattern_D

from neural_net_components import create_neural_net
from neural_net_functions import training

from test import predecirPatronesOriginales, predecirSetTest, predecirSetValidacion
# cargo el dataset
dataset = cargar_datos(100, 10)
# separo el dataset
input_X = np.array(dataset[0])
input_Y = np.array(dataset[1])
entrada_test = np.array(dataset[2])
salida_test = np.array(dataset[3])
entrada_validacion = np.array(dataset[4])
salida_validacion = np.array(dataset[5])


# Algunos patrones para probar
patronB = np.array(pattern_B).ravel()
patronD = np.array(pattern_D).ravel()
patronF = np.array(pattern_F).ravel()
patB = distortion_pattern(pattern_B, 0.3)
pat = np.array(patB)
patdist = pat.ravel()

# defino una topologia para la red
# 100 entradas, 5 o 10 neuronas en la capa oculta, 5 o 10 neuronas en la capa oculta, 3 salidas
topologia = [100, 10, 10, 3]
# tipo de datos de topologia
red_neuronal = create_neural_net(topologia, fa)
trained_neural_net = None
data_trainning = training(100, red_neuronal, input_X, input_Y,
                          entrada_validacion, salida_validacion, 0.5, 0.5, fa, 0.9, 100)

trained_neural_net = data_trainning[0]  # red neuronal entrenada
print("Red Neuronal", trained_neural_net)
print()
print("Precision", data_trainning[1], "dimension", len(
    data_trainning[1]))  # accuracy en cada epoca
print()
print("MSE", data_trainning[2], "dimension",
      len(data_trainning[2]))  # mse en cada epoca

# print()
# predecirPatronesOriginales(
#     trained_neural_net, patronB, patronD, patronF, patdist)
# print()
# predecirSetTest(trained_neural_net, entrada_test)
# print()
# predecirSetValidacion(trained_neural_net, entrada_validacion)
