
from neural_net_components import forward_pass, back_propagation

def which_letter_is(output, threshold):
    mapping = [
        ("B", output[0][0]), 
        ("F", output[0][1]), 
        ("D", output[0][2])
    ]
    
    letter_is = sorted(mapping, key=lambda x: x[1])
    # print(letter_is)
    return [letter_is[2][0], letter_is[2][1]] if float(letter_is[2][1]) > threshold else "No se pudo identificar la letra"



def train_once(neural_net, X, Y, lr, cost_f):
    outputs = forward_pass(neural_net, X)
    trained_neural_net = back_propagation(neural_net, Y, lr, cost_f, outputs)
    return trained_neural_net

# defino una funcion para entrenar la red
def training(epochs, neural_net, X, Y, lr, cost_f):
    trained_neural_net = train_once(neural_net, X, Y, lr, cost_f)
    for i in range(epochs - 1): # -1 porque lo entrena ya una vez al inicio de la funcion
        # Entrenamos a la red
        trained_neural_net = train_once(trained_neural_net, X, Y, lr, cost_f)
    return trained_neural_net

# defino una funcion para predecir
def predict(trained_neural_net, pattern):
    result = forward_pass(trained_neural_net, pattern)

    return which_letter_is(result[-1][1], 0.1)