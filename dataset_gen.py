import numpy as np
import math
import random

pattern_B = [
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '*', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '*', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '*', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '*', '*', '*', '*', '*', '-', '-', '-'],
    ['-', '-', '*', '-', '-', '-', '-', '*', '-', '-'],
    ['-', '-', '*', '-', '-', '-', '-', '*', '-', '-'],
    ['-', '-', '*', '-', '-', '-', '-', '*', '-', '-'],
    ['-', '-', '*', '*', '*', '*', '*', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
]

pattern_D = [
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '*', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '*', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '*', '-', '-'],
    ['-', '-', '-', '*', '*', '*', '*', '*', '-', '-'],
    ['-', '-', '*', '-', '-', '-', '-', '*', '-', '-'],
    ['-', '-', '*', '-', '-', '-', '-', '*', '-', '-'],
    ['-', '-', '*', '-', '-', '-', '-', '*', '-', '-'],
    ['-', '-', '-', '*', '*', '*', '*', '*', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
]

pattern_F = [
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '*', '*', '-', '-', '-'],
    ['-', '-', '-', '-', '*', '-', '-', '*', '-', '-'],
    ['-', '-', '-', '-', '*', '-', '-', '-', '-', '-'],
    ['-', '-', '*', '*', '*', '*', '*', '-', '-', '-'],
    ['-', '-', '-', '-', '*', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '*', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '*', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '*', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
]

def print_pattern(pattern):

    for i in range(len(pattern)):
        line = ''
        for j in range(len(pattern[i])):
            line += pattern[i][j]
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
    total_position_char = []
    for i in range(len(pattern)):
        for j in range(len(pattern[i])):
            position_char = []
            if(pattern[i][j] == '*'):
                count_char += 1
                position_char.append(i)
                position_char.append(j)
                total_position_char.append(position_char)
    
    num_cells_modify = math.ceil(count_char * percentage)

    for i in range(num_cells_modify):
        pos_of_total = random.randint(0, len(total_position_char) - 1)
        pos_0 = total_position_char[pos_of_total][0]
        pos_1 = total_position_char[pos_of_total][1]
        pattern[pos_0][pos_1] = '-'
        new_position = random_position_change(pos_0, pos_1)
        pattern[new_position[0]][new_position[1]] = '*'
        total_position_char.pop(pos_of_total)
    return pattern



dist_pattern = distortion_pattern(pattern_B, 0.3) # 30% de distorsion
print_pattern(dist_pattern)