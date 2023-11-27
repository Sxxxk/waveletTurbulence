import math
import numpy as np
from itertools import product

_NOISE_TILE_SIZE = 128


#************************************************
# This file is a direct translation of WAVELET_NOISE.H
# and a part of WTURBULENCE.cpp, found here :
# [https://www.cs.cornell.edu/~tedkim/WTURB/]
#************************************************

# Loads the noise tile
def load_noise_tile(filename):
    with open(filename, 'r') as file:
        noise_tile = [float(i) for i in file.readline().split(" ") if i.strip()]
    return noise_tile

def WNoiseDx(x, y, z, noise_tile):
    n = _NOISE_TILE_SIZE
    weights = [[0, 0, 0]] * 3

    mid_x = math.ceil(x - 0.5)
    t = mid_x - (x - 0.5)
    weights[0][0] = -t
    weights[0][1] = 1 - t
    weights[0][2] = 2 * t - 1

    mid_y = math.ceil(y - 0.5)
    t = mid_y - (y - 0.5)
    weights[1][0] = t * t / 2
    weights[1][1] = (1 - t) * (1 - t) / 2
    weights[1][2] = 1 - weights[1][0] - weights[1][2]

    mid_z = math.ceil(z - 0.5)
    t = mid_z - (z - 0.5)
    weights[2][0] = t * t / 2
    weights[2][1] = (1 - t) * (1 - t) / 2
    weights[2][2] = 1 - weights[2][0] - weights[2][2]

    c = [0, 0, 0]
    result = 0

    for (i, j, k) in product(range(-1, 2), repeat=3):
        weight = 1
        c[0] = (mid_x + i) % 128
        weight *= weights[0][i + 1]
        c[1] = (mid_y + j) % 128
        weight *= weights[1][j + 1]
        c[2] = (mid_z + k) % 128
        weight *= weights[2][k + 1]
        result += weight * noise_tile[c[2] * n * n + c[1] * n + c[0]]

    return result

def WNoiseDy(x, y, z, noise_tile):
    n = _NOISE_TILE_SIZE
    weights = [[0, 0, 0]] * 3

    mid_x = math.ceil(x - 0.5)
    t = mid_x - (x - 0.5)
    weights[0][0] = t * t / 2
    weights[0][1] = (1 - t) * (1 - t) / 2
    weights[0][2] = 1 - weights[0][0] - weights[0][2]

    mid_y = math.ceil(y - 0.5)
    t = mid_y - (y - 0.5)
    weights[1][0] = -t
    weights[1][1] = 1 - t
    weights[1][2] = 2 * t - 1

    mid_z = math.ceil(z - 0.5)
    t = mid_z - (z - 0.5)
    weights[2][0] = t * t / 2
    weights[2][1] = (1 - t) * (1 - t) / 2
    weights[2][2] = 1 - weights[2][0] - weights[2][2]

    c = [0, 0, 0]
    result = 0

    for (i, j, k) in product(range(-1, 2), repeat=3):
        weight = 1
        c[0] = (mid_x + i) % 128
        weight *= weights[0][i + 1]
        c[1] = (mid_y + j) % 128
        weight *= weights[1][j + 1]
        c[2] = (mid_z + k) % 128
        weight *= weights[2][k + 1]
        result += weight * noise_tile[c[2] * n * n + c[1] * n + c[0]]

    return result

def WNoiseDz(x, y, z, noise_tile):
    n = _NOISE_TILE_SIZE
    weights = [[0, 0, 0]] * 3

    mid_x = math.ceil(x - 0.5)
    t = mid_x - (x - 0.5)
    weights[0][0] = t * t / 2
    weights[0][1] = (1 - t) * (1 - t) / 2
    weights[0][2] = 1 - weights[0][0] - weights[0][2]

    mid_y = math.ceil(y - 0.5)
    t = mid_y - (y - 0.5)
    weights[1][0] = t * t / 2
    weights[1][1] = (1 - t) * (1 - t) / 2
    weights[1][2] = 1 - weights[1][0] - weights[1][2]

    mid_z = math.ceil(z - 0.5)
    t = mid_z - (z - 0.5)
    weights[2][0] = -t
    weights[2][0] = 1 - t
    weights[2][0] = 2 * t - 1

    c = [0, 0, 0]
    result = 0

    for (i, j, k) in product(range(-1, 2), repeat=3):
        weight = 1
        c[0] = (mid_x + i) % 128
        weight *= weights[0][i + 1]
        c[1] = (mid_y + j) % 128
        weight *= weights[1][j + 1]
        c[2] = (mid_z + k) % 128
        weight *= weights[2][k + 1]
        result += weight * noise_tile[c[2] * n * n + c[1] * n + c[0]]

    return result

def WVelocity(x, y, z, noise_tile):
    # Use abitrary offsets on the noise tile instead of using
    # 3 different noise tiles
    x1, y1, z1 = x + _NOISE_TILE_SIZE / 2, y, z
    x2, y2, z2 = x, y + _NOISE_TILE_SIZE / 2, z
    x3, y3, z3 = x, y, z + _NOISE_TILE_SIZE / 2

    f1y = WNoiseDy(x1, y1, z1, noise_tile)
    f1z = WNoiseDz(x1, y1, z1, noise_tile)

    f2x = WNoiseDx(x2, y2, z2, noise_tile)
    f2z = WNoiseDz(x2, y2, z2, noise_tile)

    f3x = WNoiseDx(x3, y3, z3, noise_tile)
    f3y = WNoiseDy(x3, y3, z3, noise_tile)

    return np.array([f3y - f2z, f1z - f3x, f2x - f1y])

#实现速度场的小波扰动
def turbulence(x, y, z, i_min, i_max, noise_tile):
    sum = np.array([0., 0., 0.])
    for i in range(i_min, i_max + 1):
        sum += WVelocity(2**i * x, 2**i * y, 2**i * z, noise_tile) \
               * 2**(-(5/6) * (i - i_min))
    
    return sum
