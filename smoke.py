#!/usr/bin/env python3

import pyopenvdb as vdb
from interpolate import trilinear_interpolation

class Smoke:
    def __init__(self, filename):
        self.density_grid = None
        self.density_accessor = None
        self.velocity_grid = None
        self.velocity_accessor = None

        self.res_x = 0
        self.res_y = 0
        self.res_z = 0

        grids = vdb.readAllGridMetadata(filename)
        for grid in grids:
            if grid.name == 'density':
                self.density_grid = grid
                self.density_accessor = grid.getAccessor()
            elif grid.name == 'v':
                self.velocity_grid = grid
                self.velocity_accessor = grid.getAccessor()

        if (self.density_grid is None or self.velocity_grid is None):
            raise Exception("Given file misses density or velocity grid.")
        

smoke = Smoke("smoke2.vdb")
