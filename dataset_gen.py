from email.policy import strict
import re
import numpy as np
import math
import random

from yaml import unsafe_load

pattern_B = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

pattern_D = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

pattern_F = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]


def print_pattern(pattern):

    for i in range(len(pattern)):
        line = ''
        for j in range(len(pattern[i])):
            line += str(pattern[i][j])
        print(line)


def random_position_change(pos1, pos2):
    array = [
        [pos1 + 1, pos2],
        [pos1 - 1, pos2],
        [pos1, pos2 + 1],
        [pos1, pos2 - 1]
    ]
    r = random.randint(0, 3)
    return array[r]


def distortion_pattern(pattern, percentage):
    """
        percentage: este valor tiene que ir de 0 a 1 indicando el porcentaje de distorsion
    """
    count_char = 0
    total_position_to_modify = []
    for i in range(len(pattern) - 1):
        for j in range(len(pattern[i]) - 1):
            position_char = []
            if (pattern[i][j] == 1):
                count_char += 1
                position_char.append(i)
                position_char.append(j)
                total_position_to_modify.append(position_char)

    num_cells_modify = math.ceil(count_char * percentage)

    for i in range(num_cells_modify):
        pos_of_total = random.randint(0, len(total_position_to_modify) - 1)
        pos_0 = total_position_to_modify[pos_of_total][0]
        pos_1 = total_position_to_modify[pos_of_total][1]
        pattern[pos_0][pos_1] = 0
        new_position = random_position_change(pos_0, pos_1)
        pattern[new_position[0]][new_position[1]] = 1
        total_position_to_modify.pop(pos_of_total)
    return pattern


def random_distortion_set(a, b):
    rand = random.randrange(a, b)
    if (rand < 10):
        return float('0.0' + str(rand))
    else:
        return float('0.' + str(rand))


def create_dataset(n_ejemplos, arrayPattern):
    porcentaje_entrenamineto = round(n_ejemplos * 0.6)
    porcentaje_entrenamineto_sin_distorsion = round(
        porcentaje_entrenamineto * 0.1)
    porcentaje_entrenamineto_con_distorsion = round(
        porcentaje_entrenamineto * 0.9)
    porcentaje_prueba = round(n_ejemplos * 0.1)
    porcentaje_validacion = round(n_ejemplos * 0.3)
    # creo el dataset de entrenamiento representativo del 70% de los datos
    array_dist_data = []
    array_real_data = []
    array_data_test = []
    array_data_validation = []
    # 10% de los datos de entrenamiento sin distorsion
    # Caracter B
    for i in range(round(porcentaje_entrenamineto_sin_distorsion*0.33)):
        dist_pattern = arrayPattern[0]
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_dist_data.append(dist_pattern)
        array_real_data.append([1, 0, 0])
    # Caracter F
    for i in range(round(porcentaje_entrenamineto_sin_distorsion*0.33)):
        dist_pattern = arrayPattern[1]
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_dist_data.append(dist_pattern)
        array_real_data.append([0, 1, 0])
    # Caracter D
    for i in range(round(porcentaje_entrenamineto_sin_distorsion*0.33)):
        dist_pattern = arrayPattern[2]
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_dist_data.append(dist_pattern)
        array_real_data.append([0, 0, 1])
    # 90% de los datos de entrenamiento con distorsion
    # Caracter B
    for i in range(round(porcentaje_entrenamineto_con_distorsion*0.33)):
        dist_pattern = distortion_pattern(
            arrayPattern[0], random_distortion_set(0, 30))
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_dist_data.append(dist_pattern)
        array_real_data.append([1, 0, 0])
    # Caracter F
    for i in range(round(porcentaje_entrenamineto_con_distorsion*0.33)):
        dist_pattern = distortion_pattern(
            arrayPattern[1], random_distortion_set(0, 30))
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_dist_data.append(dist_pattern)
        array_real_data.append([0, 1, 0])
    # Caracter D
    for i in range(round(porcentaje_entrenamineto_con_distorsion*0.33)):
        dist_pattern = distortion_pattern(
            arrayPattern[2], random_distortion_set(0, 30))
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_dist_data.append(dist_pattern)
        array_real_data.append([0, 0, 1])
    # Datos de prueba
    # Caracter B
    for i in range(round(porcentaje_prueba*0.33)):
        dist_pattern = distortion_pattern(
            arrayPattern[0], random_distortion_set(0, 30))
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_data_test.append(dist_pattern)
    # Caracter F
    for i in range(round(porcentaje_prueba*0.33)):
        dist_pattern = distortion_pattern(
            arrayPattern[1], random_distortion_set(0, 30))
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_data_test.append(dist_pattern)
    # Caracter D
    for i in range(round(porcentaje_prueba*0.33)):
        dist_pattern = distortion_pattern(
            arrayPattern[2], random_distortion_set(0, 30))
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_data_test.append(dist_pattern)
    # Datos de validacion
    # Caracter B
    for i in range(round(porcentaje_validacion*0.33)):
        dist_pattern = distortion_pattern(
            arrayPattern[0], random_distortion_set(0, 30))
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_data_validation.append(dist_pattern)
    # Caracter F
    for i in range(round(porcentaje_validacion*0.33)):
        dist_pattern = distortion_pattern(
            arrayPattern[1], random_distortion_set(0, 30))
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_data_validation.append(dist_pattern)
    # Caracter D
    for i in range(round(porcentaje_validacion*0.33)):
        dist_pattern = distortion_pattern(
            arrayPattern[2], random_distortion_set(0, 30))
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_data_validation.append(dist_pattern)
    return [array_dist_data, array_real_data, array_data_test, array_data_validation]


def guardar_datos():
    datos = create_dataset(1000, [pattern_B, pattern_F, pattern_D])
    datos_entrada = np.array(datos[0])
    datos_salida = np.array(datos[1])
    datos_prueba = np.array(datos[2])
    datos_validacion = np.array(datos[3])
    np.savetxt(
        "datasets/1000/30_validacion/entrenamiento/entrada.txt", datos_entrada)
    np.savetxt(
        "datasets/1000/30_validacion/entrenamiento/salida.txt", datos_salida)
    np.savetxt(
        "datasets/1000/30_validacion/test/test.txt", datos_prueba)
    np.savetxt(
        "datasets/1000/30_validacion/validacion/validacion.txt", datos_validacion)


def cargar_datos(n_ejemplos, porcentaje_validacion):
    datos_entrada = np.loadtxt(
        "datasets/"+str(n_ejemplos)+"/"+str(porcentaje_validacion)+"_validacion/entrenamiento/entrada.txt")
    datos_salida = np.loadtxt(
        "datasets/"+str(n_ejemplos)+"/"+str(porcentaje_validacion)+"_validacion/entrenamiento/salida.txt")
    datos_prueba = np.loadtxt("datasets/"+str(n_ejemplos)+"/" +
                              str(porcentaje_validacion)+"_validacion/test/test.txt")
    datos_validacion = np.loadtxt(
        "datasets/"+str(n_ejemplos)+"/"+str(porcentaje_validacion)+"_validacion/validacion/validacion.txt")
    return [datos_entrada, datos_salida, datos_prueba, datos_validacion]


guardar_datos()
