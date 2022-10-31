import numpy as np
from dataset_gen import cargar_datos


# funcion para armar la arquitectura de la red
def crear_arquitectura(opciones):
    n_capas = int(opciones[1])
    n_neuronas = int(opciones[2])
    artquitecura = np.zeros(n_capas + 2, dtype=int)
    for i in range(len(artquitecura)):
        if i == 0:
            artquitecura[i] = 100
        elif i == len(artquitecura) - 1:
            artquitecura[i] = 3
        else:
            artquitecura[i] = n_neuronas
    return list(artquitecura)


# funcion para obtener el momentum
def getMomentum(opciones):
    return float(opciones[-1])


# funcion para obtener el dataset
# 100_10validacion  	500_10validacion	1000_10validacion
# 100_20validacion	500_20validacion	1000_20validacion
# 100_30validacion	500_30validacion	1000_30validacion
def getDataset(opciones):
    # mapa de datasets
    datasets = {
        '100_10validacion': (100, 10),
        '100_20validacion': (100, 20),
        '100_30validacion': (100, 30),
        '500_10validacion': (500, 10),
        '500_20validacion': (500, 20),
        '500_30validacion': (500, 30),
        '1000_10validacion': (1000, 10),
        '1000_20validacion': (1000, 20),
        '1000_30validacion': (1000, 30)
    }
    # si el dataset que recibo es igual a la clave del mapa retorno el valor
    for key in datasets:
        if key == opciones[0]:
            data = datasets[key]
    dataset = cargar_datos(data[0], data[1])
    return dataset


# funcion para tratar ejemplo de entrada
def tratarEntrada(arreglo):
    patron = np.array(arreglo).ravel()
    return patron
