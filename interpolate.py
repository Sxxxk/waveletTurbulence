import pyopenvdb as vdb
import numpy as np

def lerp(a, b, t):
    return (a + t * (b - a))

def trilinear_interpolation(grid_accessor, x, y, z):
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    z0 = int(z)
    z1 = z0 + 1

    xd = x - x0
    yd = y - y0
    zd = z - z0
    
    c000 = grid_accessor.probeValue((x0, y0, z0))[0]
    c001 = grid_accessor.probeValue((x0, y0, z1))[0]
    c010 = grid_accessor.probeValue((x0, y1, z0))[0]
    c011 = grid_accessor.probeValue((x0, y1, z1))[0]
    c100 = grid_accessor.probeValue((x1, y0, z0))[0]
    c101 = grid_accessor.probeValue((x1, y0, z1))[0]
    c110 = grid_accessor.probeValue((x1, y1, z0))[0]
    c111 = grid_accessor.probeValue((x1, y1, z1))[0]

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
