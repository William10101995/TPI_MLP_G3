from multiprocessing.dummy import Array
import numpy as np
import math
import random
import os

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
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

pattern_F = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 3, 3, 3, 3, 3, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
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
            if(pattern[i][j] == 1):
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


# dist_pattern = distortion_pattern(pattern_B, 0.3) # 30% de distorsion
# print_pattern(dist_pattern)

# def create_correct_format(dist_pattern):
#     X = []
#     Y = []
#     for i in range(len(dist_pattern)):
#         for j in range(len(dist_pattern[i])):
#             X.append([i, j])                    # Guarda las posiciones
#             Y.append([dist_pattern[i][j]])        # Guarda la clase a la que pertenece la posicion anterior
    
#     return [X, Y]                               #[[[i, j], ...], [0, 1, ...]] la salida es asi


def random_distortion_set(a, b):
    rand = random.randrange(a, b)
    if(rand < 10):
        return  float('0.0' + str(rand))
    else:
        return float('0.' + str(rand))


def which_letter(pattern):
    #Este if esta hecho para probar si anda o no la idea, puede mejorarse mucho
    if(1 in pattern):
        return 1
    elif(2 in pattern):
        return 2
    else:
        return 3


def create_dataset(n, pattern_array):
    array_dist_data = []
    array_real_data = []
    for i in range(n):
        pattern = pattern_array[random.randint(0, 2)]
        dist_pattern = distortion_pattern(pattern, random_distortion_set(0, 30))
        dist_pattern = np.array(dist_pattern)
        dist_pattern = dist_pattern.ravel()
        array_dist_data.append(dist_pattern)    # agrego el patron distorsionado
        array_real_data.append(which_letter(dist_pattern))                # agrego el valor real que representa
    return [np.array(array_dist_data), np.array(array_real_data)]

test = create_dataset(2, [pattern_B, pattern_D, pattern_F])
print(np.array(test[0]))
print(np.array(test[1])[:, np.newaxis])







































# def create_complete_dataset(n, pattern_array):
    
#     # file_X = open("X.txt", "w")
#     # file_Y = open("Y.txt", "w")

#     array_X = []
#     array_Y = []
#     for i in range(n):
#         pattern = pattern_array[random.randint(0, 2)]
#         distorion = random_distortion_set(0, 31)
#         dist_pattern = distortion_pattern(pattern, distorion)
#         correct_format = create_correct_format(dist_pattern)
#         array_X.append(correct_format[0])
#         array_Y.append(correct_format[1])

#     #     file_X.write(str(correct_format[0]) + os.linesep)
#     #     file_Y.write(str(correct_format[1]) + os.linesep)
    
#     # file_X.close()
#     # file_Y.close()
#     return [array_X, array_Y]

# # create_complete_dataset(2, [pattern_B, pattern_D, pattern_F])


# # dist_pattern = distortion_pattern(pattern_B, 0.3) # 30% de distorsion
# # # # print_pattern(dist_pattern)

# # print(create_correct_format(dist_pattern)[0])

# def print_pattern_from_correct_form(pattern):
#     for k in range(len(pattern[1])):
#         array = []
#         for i in range(10):
#             array.append([None, None, None, None, None, None, None, None, None, None])

#         for i in range(10):
#             for j in range(10):
#                 array[i][j] = pattern[1][k][int(str(i) + str(j))][0]

#         print_pattern(array);
#         print("\n")
