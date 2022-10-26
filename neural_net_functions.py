
from neural_net_components import forward_pass, back_propagation
import activation_functions as fa
def train_once(neural_net, X, Y, lr, momentum, der_cost_f):
    salidas = forward_pass(neural_net, X)
    trained_neural_net = back_propagation(neural_net, Y, lr, momentum, der_cost_f, salidas)
    return trained_neural_net

def training(epochs, neural_net, input_X, input_Y, lr, momentum, der_cost_f):
    trained_neural_net = train_once(neural_net, input_X, input_Y, lr, momentum, der_cost_f)
    for i in range(epochs):
        trained_neural_net = train_once(trained_neural_net, input_X, input_Y, lr, momentum, der_cost_f)
    return trained_neural_net

def predict(trained_neural_net, pattern, threshold):
    result = forward_pass(trained_neural_net, pattern)
    return which_letter_is(result[-1][1], threshold)

def which_letter_is(output, threshold):
    mapping = [
        ("B", output[0][0]), 
        ("F", output[0][1]), 
        ("D", output[0][2])
    ]
    
    letter_is = sorted(mapping, key=lambda x: x[1])
    return [letter_is[2][0], letter_is[2][1]] if float(letter_is[2][1]) > threshold else "No se pudo identificar la letra"


