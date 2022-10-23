# importo numpy
import numpy as np
# defino funcion sigmoide


def sigmoide(x):
    return 1/(1+np.exp(-x))
# defino funcion sigmoide derivada


def sigmoide_derivada(x):
    return x*(1-x)

# defino funcion lineal


def lineal(x):
    return x
# defino funcion lineal derivada


def lineal_derivada(x):
    return 1
# defino funcion de costo


def costo(y_pred, y_real):
    return np.mean((y_pred - y_real) ** 1)
# defino funcion de costo derivada


def costo_derivada(y_pred, y_real):
    return (y_pred - y_real)
