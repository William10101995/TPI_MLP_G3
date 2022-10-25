# importo numpy
import numpy as np

# defino funcion sigmoide
def sigmoide(x):
    return 1/(1+np.exp(-x))

# defino funcion sigmoide derivada
def sigmoide_derivative(x):
    return x*(1-x)

# defino funcion lineal
def lineal(x):
    return 0.1*x

# defino funcion lineal derivada
def lineal_derivative(x):
    return 0.1

# defino funcion de costo
def cost(y_pred, y_real):
    return np.mean((y_pred - y_real) ** 1)

# defino funcion de costo derivada
def derived_cost(y_pred, y_real):
    return (y_pred - y_real)
