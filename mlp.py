# importo numpy
import numpy as np
# importo funciones de activacion
import funciones_act as fa
from dataset_gen import cargar_datos, distortion_pattern, pattern_F, pattern_B, pattern_D

from neural_net_components import crear_red, forward_pass, back_propagation
from neural_net_functions import training, predict

from test import predecirPatronesOriginales, predecirSetTest, predecirSetValidacion
# cargo el dataset
dataset = cargar_datos(500, 10)
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
red_neuronal = crear_red(topologia, fa)
trained_neural_net = None
trained_neural_net = training(
    100000, red_neuronal, input_X, input_Y, 0.5, 0.5, fa)


predecirPatronesOriginales(
    trained_neural_net, patronB, patronD, patronF, patdist)
print()
predecirSetTest(trained_neural_net, entrada_test)
print()
predecirSetValidacion(trained_neural_net, entrada_validacion)
