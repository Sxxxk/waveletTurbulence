#!/usr/bin/env python3

import pyopenvdb as vdb
from interpolate import trilinear_interpolation
from itertools import product

from time import time

class Smoke:
    def __init__(self, filename):
        self.density_grid = None
        self.density_accessor = None
        self.velocity_grid = None
        self.velocity_accessor = None

        grids = vdb.readAllGridMetadata(filename)
        for grid in grids:
            print("Reading grid '%s'" % grid.name)
            if grid.name == 'density':
                self.density_grid = vdb.read(filename, grid.name)
                self.density_accessor = self.density_grid.getAccessor()
            elif grid.name == 'v':
                self.velocity_grid = vdb.read(filename, grid.name)
                self.velocity_accessor = self.velocity_grid.getAccessor()

        if (self.density_grid is None or self.velocity_grid is None):
            raise Exception("Given file misses density or velocity grid.")
        
        # We get the max active index: + 1 because we want to iterate until there
        # but python for loops stops at max - 1
        #                              + 1 because we also take the empty voxel just
        # after the last one.
        # Thus the +2
        self.n = max(self.density_grid.evalActiveVoxelBoundingBox()[1]) + 2
        print("Base resolution : %s." % self.n)
        print("Grids opened.")

    def make_higher_res(self, N, filename):
        N_density_grid = vdb.FloatGrid()
        N_velocity_grid = vdb.Vec3SGrid()
        N_density_accessor = N_density_grid.getAccessor()
        N_velocity_accessor = N_velocity_grid.getAccessor()

        scale = N / self.n

        print("Starting enhancing to new resolution %s." % N)

        t0 = time()

        loop = 0
        max_loop = N**3 - 1
        percent = int(max_loop / 100)

        for (i, j, k) in product(range(N), repeat=3):
            if (loop % percent == 0 or loop == max_loop):
                printProgressBar(loop, max_loop, t0)
            
            x = i / scale            
            y = j / scale
            z = k / scale

            density = trilinear_interpolation(self.density_accessor, x, y, z)
            N_density_accessor.setValueOn((i, j, k), density)

            velocity = trilinear_interpolation(self.velocity_accessor, x, y, z)
            N_velocity_accessor.setValueOn((i, j, k), velocity)

            # print("%s %s %s: density: %s  velocity: %s" % (i, j, k, density, velocity))

            loop += 1

        print("total time: ", time() - t0, "s.")
        print("saving grids in %s." % filename)
        vdb.write(filename, grids = [N_density_grid, N_velocity_grid])


# Print iterations progress
# Taken from [https://stackoverflow.com/a/34325723]
def printProgressBar (iteration, total, start_time, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        start_time  - Required  : starting time (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    eta = "-"
    if iteration != 0:
        eta = ("{0:." + str(decimals) + "f}").format(((time() - start_time) / iteration) * (total - iteration))
    print(f'\r{prefix} |{bar}| {percent}% {suffix} ETA: {eta}s.        ', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


smoke = Smoke("test.vdb")
smoke.make_higher_res(80, "test_enhanced.vdb")
