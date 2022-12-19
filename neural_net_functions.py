import numpy as np
from neural_net_components import back_propagation, forward_pass


# Funcion para obtener la precision de la red a lo largo de las epocas
def getAccuracy(neural_net, X, Y, threshold):
    correct = 0
    for x, y in zip(X, Y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        predict = forward_pass(neural_net, x)
        predicted_letter = which_letter_is(predict[-1][1], threshold)
        real_letter = which_letter_is(y, 0.99)
        if predicted_letter[0] == real_letter[0]:
            correct += 1

    accuracy = round((correct/len(Y)) * 100, 5)
    return accuracy


# Funcion para obtener el error cuadratico medio del conjunto de validacion a lo largo de las epocas
def getMSE(neural_net, X, Y, threshold):
    mse = 0
    for x, y in zip(X, Y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        predict = forward_pass(neural_net, x)
        predicted_letter = which_letter_is(predict[-1][1], threshold)
        real_letter = which_letter_is(y, 0.99)

        mse += (predicted_letter[1] - real_letter[1])**2
    mse = round(mse / len(Y), 5)
    return mse

def getValidatonAndTestError(neural_net, X, Y, threshold):
    error = 0
    for x, y in zip(X, Y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        predict = forward_pass(neural_net, x)
        predicted_letter = which_letter_is(predict[-1][1], threshold)
        real_letter = which_letter_is(y, 0.99)

        error += ((real_letter[1] - predicted_letter[1])**2)/2
    error = round(error / len(Y), 5)
    return error

# Funcion para entrenar una sola vez a la red
def train_once(neural_net, X, Y, lr, momentum, cost_f):
    output = forward_pass(neural_net, X)
    neural_net = back_propagation(neural_net, Y, lr, momentum, cost_f, output)
    return neural_net


# Funcion para entrenar la red una cierta cantidad de epocas
def training(epochs, neural_net, X, Y, X_val, Y_val, X_test, Y_test, lr, momentum, fa, threshold, desired_accuracy):
    x1 = X[0]
    y1 = Y[0]
    # Convierto el primer patron de entrenamiento a un array de 2 dimensiones
    # Entrada, esto seria el patron
    x1 = np.atleast_2d(x1)
    # Salida deseada para el patron
    y1 = np.atleast_2d(y1)
    # Inicializo las listas para guardar los datos de accuracy y mse a lo largo de las epocas
    final_accuracy = []
    final_mse = []
    final_training_error = []
    final_validating_error = []
    # Entrenamos a la red por primera vez
    trained_neural_net = train_once(neural_net, x1, y1, lr, momentum, fa.costo_derivada)
    # Entrenamos a la red por las epocas restantes
    for epoch in range(epochs):
        # Entrenamos a la red por cada ejemplo del conjunto de entrenamiento
        for x, y in zip(X, Y):
            # Convertimos los ejemplos a matrices de 2 dimensiones para poder operar con ellos
            # Entrada, esto seria el patron
            x = np.atleast_2d(x)
            # Salida deseada para el patron
            y = np.atleast_2d(y)

            trained_neural_net = train_once(
                trained_neural_net, x, y, lr, momentum, fa.costo_derivada)
        # Obtengo la precision de la red
        accuracy = getAccuracy(trained_neural_net, X, Y, threshold)
        # Obtengo el error cuadratico medio del conjunto de validacion
        mse = getMSE(trained_neural_net, X_val, Y_val, threshold)
        training_error = getValidatonAndTestError(trained_neural_net, X, Y, threshold)
        validating_error = getValidatonAndTestError(trained_neural_net, X_val, Y_val, threshold)
        # Guardo los datos de accuracy y mse
        final_accuracy.append([epoch, accuracy])
        final_mse.append([epoch, mse])

        final_training_error.append([epoch, training_error])
        final_validating_error.append([epoch, validating_error])

        if (accuracy >= desired_accuracy):
            break
    # Obtenemos el error cuadratico medio del conjunto de test luego de entrenar
    mseTest = getMSE(trained_neural_net, X_test, Y_test, threshold)
    return [trained_neural_net, final_accuracy, final_mse, mseTest, final_training_error, final_validating_error]


# Funcion para predecir un ejemplo
def predict(trained_neural_net, pattern):
    result = forward_pass(trained_neural_net, pattern)
    return which_letter_is(result[-1][1], 0.5)


# Funcion para obtener la letra a la que pertenece un patron
def which_letter_is(output, threshold):
    mapping = [
        ("B", output[0][0]),
        ("F", output[0][1]),
        ("D", output[0][2])
    ]

    letter_is = sorted(mapping, key=lambda x: x[1])
    return [letter_is[2][0], letter_is[2][1], letter_is[1][0], letter_is[1][1], letter_is[0][0], letter_is[0][1]] if float(letter_is[2][1]) > threshold else ["No se pudo identificar la letra", 0]
