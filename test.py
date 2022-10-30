from neural_net_functions import predict
# Funciones de prueba
# Funcion para predecir patrones originales y un distorsionado
def predecirPatronesOriginales(trained_neural_net, patB, patD, patF, patDist):
    print("Prediciendo patrones originales")
    print("Patron B: ", predict(trained_neural_net, patB)[0])
    print("Patron F: ", predict(trained_neural_net, patF)[0])
    print("Patron D: ", predict(trained_neural_net, patD)[0])
    print("Prediccion B Distorsionado 30%: ", predict(trained_neural_net, patDist)[0])


# Funcion para predecir set de test
def predecirSetTest(trained_neural_net, test):
    print("Prediciendo set de test")
    for i in range(len(test)):
        print("Patron ", i, ": ", predict(trained_neural_net, test[i])[0])


# Funcion para predecir set de validacion
def predecirSetValidacion(trained_neural_net, val):
    print("Prediciendo set de validacion")
    for i in range(len(val)):
        print("Patron ", i, ": ", predict(trained_neural_net, val[i])[0])
