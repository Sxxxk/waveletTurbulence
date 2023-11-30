import pyopenvdb as vdb
import numpy as np

def lerp(a, b, t):
    return (a + t * (b - a))

def trilinear_interpolation(grid, x, y, z):
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    z0 = int(z)
    z1 = z0 + 1

    xd = x - x0
    yd = y - y0
    zd = z - z0

    #三线性插值。由于得到的xyz不会处在网格的边界上。因此需要周围8个坐标上的值进行插值得到
    try:
        c000 = grid.probeValue((x0, y0, z0))[0]
        c001 = grid.probeValue((x0, y0, z1))[0]
        c010 = grid.probeValue((x0, y1, z0))[0]
        c011 = grid.probeValue((x0, y1, z1))[0]
        c100 = grid.probeValue((x1, y0, z0))[0]
        c101 = grid.probeValue((x1, y0, z1))[0]
        c110 = grid.probeValue((x1, y1, z0))[0]
        c111 = grid.probeValue((x1, y1, z1))[0]
    except AttributeError:
        raise Exception("Invalid grid type in interpolation : %s" % type(grid))

    if type(c000) == tuple:
        c000 = np.array(c000)
        c001 = np.array(c001)
        c010 = np.array(c010)
        c011 = np.array(c011)
        c100 = np.array(c100)
        c101 = np.array(c101)
        c110 = np.array(c110)
        c111 = np.array(c111)

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    return c0 * (1 - zd) + c1 * zd

#小波能量场的插值上采样，同样使用的是三线性上采样
def interpolate_wl(grids, x, y, z):
#将采样坐标再缩小1/2
    x = x/2 - 0.5
    y = y/2 - 0.5
    z = z/2 - 0.5
    #得到网格分辨率
    max_grid_index = len(grids['aaa'][0][0]) - 1

    #如果采样坐标超出限制
    if x < 0:
        x = 0
    elif x >= max_grid_index:
        x = max_grid_index - 1

    if y < 0:
        y = 0
    elif y >= max_grid_index:
        y = max_grid_index - 1

    if z < 0:
        z = 0
    elif z >= max_grid_index:
        z = max_grid_index - 1
    #得到采样点周围的八个网格（能量值）
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    z0 = int(z)
    z1 = z0 + 1

    xd = x - x0
    yd = y - y0
    zd = z - z0

    c000 = 0
    c001 = 0
    c010 = 0
    c011 = 0
    c100 = 0
    c101 = 0
    c110 = 0
    c111 = 0
    #对于各个通道（除了aaa高通通道以外）
    for grid_name in grids:
        if grid_name != 'aaa':
            c000 += grids[grid_name][x0][y0][z0]
            c001 += grids[grid_name][x0][y0][z1]
            c010 += grids[grid_name][x0][y1][z0]
            c011 += grids[grid_name][x0][y1][z1]
            c100 += grids[grid_name][x1][y0][z0]
            c101 += grids[grid_name][x1][y0][z1]
            c110 += grids[grid_name][x1][y1][z0]
            c111 += grids[grid_name][x1][y1][z1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    return c0 * (1 - zd) + c1 * zd
