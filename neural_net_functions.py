import numpy as np
from sklearn.metrics import accuracy_score
from neural_net_components import back_propagation, forward_pass

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

    accuracy = (correct/len(Y)) * 100
    return accuracy


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

def train_once(neural_net, X, Y, lr, momentum, cost_f):

    output = forward_pass(neural_net, X)
    neural_net = back_propagation(neural_net, Y, lr, momentum, cost_f, output)
    return neural_net


def training(epochs, neural_net, X, Y, X_val, Y_val, lr, momentum, fa, threshold, desired_accuracy):
    x1 = X[0]
    y1 = Y[0]

    x1 = np.atleast_2d(x1)
    y1 = np.atleast_2d(y1)

    final_accuracy = []
    final_mse = []
    # Entrenamos a la red!
    trained_neural_net = train_once(neural_net, x1, y1, lr, momentum, fa.costo_derivada)
    for epoch in range(epochs):
        for x, y in zip(X, Y):
            x = np.atleast_2d(x)
            y = np.atleast_2d(y)

            trained_neural_net = train_once(trained_neural_net, x, y, lr, momentum, fa.costo_derivada)

        accuracy = getAccuracy(trained_neural_net, X, Y, threshold)
        mse = getMSE(trained_neural_net, X_val, Y_val, threshold)

        final_accuracy.append([epoch, accuracy])
        final_mse.append([epoch, mse])

        if(accuracy >= desired_accuracy):
            break
    return [trained_neural_net, final_accuracy, final_mse]



def predict(trained_neural_net, pattern):
    # print(trained_neural_net)
    result = forward_pass(trained_neural_net, pattern)
    return which_letter_is(result[-1][1], 0.5)

def which_letter_is(output, threshold):
    mapping = [
        ("B", output[0][0]), 
        ("F", output[0][1]), 
        ("D", output[0][2])
    ]
    
    letter_is = sorted(mapping, key=lambda x: x[1])
    return [letter_is[2][0], letter_is[2][1]] if float(letter_is[2][1]) > threshold else ["No se pudo identificar la letra", 0]