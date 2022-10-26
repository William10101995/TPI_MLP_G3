# importo numpy
import numpy as np
# importo funciones de activacion
import activation_functions as fa
from dataset_gen import cargar_datos, distortion_pattern, pattern_F, pattern_B, pattern_D
from neural_net_components import create_neural_net
from neural_net_functions import training, predict


# cargo el dataset
dataset = cargar_datos()
# separo el dataset
input_X = np.array(dataset[0], dtype=np.float64)
input_Y = np.array(dataset[1], dtype=np.float64)
test = np.array(dataset[2])
val = np.array(dataset[3])

# Algunos patrones para probar
patronB = np.array(pattern_B).ravel()
patronD = np.array(pattern_D).ravel()
patronF = np.array(pattern_F).ravel()
patD = distortion_pattern(pattern_D, 0.90)
pat = np.array(patD)
patdist = pat.ravel()


# defino una topologia para la red
# 1 entradas, 4 neuronas en la capa oculta, 4 neuronas en la capa oculta, 3 salidas
topologia = [100, 5, 5, 3]
neural_net = create_neural_net(topologia, fa)
trained_neural_net = training(10000, neural_net, input_X, input_Y, 0.05, fa.derived_cost)

# predecimos los ejemplos de test
for i in range(10):
    prediccion = predict(trained_neural_net, patdist)
    print(prediccion)
