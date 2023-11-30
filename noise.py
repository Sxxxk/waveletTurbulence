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
    weights[0][2] = 1 - t
    weights[0][1] = 2 * t - 1

    mid_y = math.ceil(y - 0.5)
    t = mid_y - (y - 0.5)
    weights[1][0] = t * t / 2
    weights[1][2] = (1 - t) * (1 - t) / 2
    weights[1][1] = 1 - weights[1][0] - weights[1][2]

    mid_z = math.ceil(z - 0.5)
    t = mid_z - (z - 0.5)
    weights[2][0] = t * t / 2
    weights[2][2] = (1 - t) * (1 - t) / 2
    weights[2][1] = 1 - weights[2][0] - weights[2][2]

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
    weights[0][2] = (1 - t) * (1 - t) / 2
    weights[0][1] = 1 - weights[0][0] - weights[0][2]

    mid_y = math.ceil(y - 0.5)
    t = mid_y - (y - 0.5)
    weights[1][0] = -t
    weights[1][2] = 1 - t
    weights[1][1] = 2 * t - 1

    mid_z = math.ceil(z - 0.5)
    t = mid_z - (z - 0.5)
    weights[2][0] = t * t / 2
    weights[2][2] = (1 - t) * (1 - t) / 2
    weights[2][1] = 1 - weights[2][0] - weights[2][2]

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

#噪声实际上是一个128*128*128的三维纹理。现在要做的就是对这个纹理进行采样
def WNoiseDz(x, y, z, noise_tile):
    #得到噪声纹理的分辨率n
    n = _NOISE_TILE_SIZE
    #初始化权重，为一个列表，其中有三个列表，均为0，0，0
    weights = [[0, 0, 0]] * 3

    #计算x的权重
    #mid_x 是不小于 x - 0.5 的最小整数
    mid_x = math.ceil(x - 0.5)
    # t = mid_x - (x - 0.5)，得到了一个介于 0 和 1 之间的值 t，用于在插值中确定权重
    # t 的计算实际上是找到 x 与 mid_x 之间的偏移，以确定 x 在这个区间内的位置
    t = mid_x - (x - 0.5)
    #通过 t 的值计算权重数组 weights[0]，这个权重数组用于进行三次插值。
    #这里的插值采用的是三次 B-样条插值的基函数，即 t^2/2，(1-t)^2/2，1-t-t^2/2
    weights[0][0] = t * t / 2
    weights[0][2] = (1 - t) * (1 - t) / 2
    weights[0][1] = 1 - weights[0][0] - weights[0][2]

    mid_y = math.ceil(y - 0.5)
    t = mid_y - (y - 0.5)
    weights[1][0] = t * t / 2
    weights[1][2] = (1 - t) * (1 - t) / 2
    weights[1][1] = 1 - weights[1][0] - weights[1][2]

    mid_z = math.ceil(z - 0.5)
    t = mid_z - (z - 0.5)
    weights[2][0] = -t
    weights[2][2] = 1 - t
    weights[2][1] = 2 * t - 1

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
    # αw1/αy
    f1y = WNoiseDy(x1, y1, z1, noise_tile)
    f1z = WNoiseDz(x1, y1, z1, noise_tile)

    f2x = WNoiseDx(x2, y2, z2, noise_tile)
    f2z = WNoiseDz(x2, y2, z2, noise_tile)

    f3x = WNoiseDx(x3, y3, z3, noise_tile)
    #αw1/αy
    f3y = WNoiseDy(x3, y3, z3, noise_tile)

    return np.array([f3y - f2z, f1z - f3x, f2x - f1y])

#用噪声合成湍流值
#传入的值分别是：低分辨率下的坐标xyz。imin,imax(详见4.11式)。噪声数据（在smoke.py中已经调用load_noise_tile进行载入了）
def turbulence(x, y, z, i_min, i_max, noise_tile):
    sum = np.array([0., 0., 0.])
    for i in range(i_min, i_max + 1):
        #WVelocity即为论文中的w，下一行即为2^[(-5/6)(i-imin)]
        sum += WVelocity(2**i * x, 2**i * y, 2**i * z, noise_tile) \
               * 2**(-(5/6) * (i - i_min))
    
    return sum
