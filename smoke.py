#!/usr/bin/env python3

import pyopenvdb as vdb
import math
import pywt
import numpy as np
from itertools import product
from time import time

from interpolate import trilinear_interpolation
from noise import load_noise_tile, turbulence


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
        bbox = self.density_grid.evalActiveVoxelBoundingBox()
        self.min_voxel = min(bbox[0]) - 1
        self.n = max(bbox[1]) + 2
        print("Base resolution : %s." % self.n)
        print("Grids opened.")

    def compute_energies(self):
        energies = np.zeros((self.n, self.n, self.n))

        for (x, y, z) in product(range(self.n), repeat=3):
            vx, vy, vz = self.velocity_accessor.probeValue((x, y, z))[0]
            energies[x, y, z] = 0.5 * (vx * vx + vy * vy + vz * vz)

        return energies

    def wl_transform(self, energies):
        wl_transform = np.zeros((self.n, self.n, self.n))

        #TODO: compute wl transform

        return wl_transform

    def make_higher_res(self, N, filename):
        # Initializing output grids
        N_density_grid = vdb.FloatGrid()
        N_velocity_grid = vdb.Vec3SGrid()
        N_density_accessor = N_density_grid.getAccessor()
        N_velocity_accessor = N_velocity_grid.getAccessor()

        # Initalizing necessary parameters
        scale = N / self.n
        min_voxel_enhanced = int(scale * self.min_voxel)

        i_min = math.ceil(math.log(self.n, 2))
        i_max = math.floor(math.log(N / 2, 2))

        energies = self.compute_energies()
        energies_wl_transform = self.wl_transform(energies)

        print("Loading noise tile.")
        noise_tile = load_noise_tile('noiseTile')
        print("Noise tile loaded.")

        print("Starting enhancing to new resolution %s." % N)

        # Utilitary variables used for the progress bar display
        t0 = time()
        loop = 0
        max_loop = (N - min_voxel_enhanced)**3 - 1
        percent = int(max_loop / 100)

        for (i, j, k) in product(range(min_voxel_enhanced, N), repeat=3):
            if (loop % percent == 0 or loop == max_loop):
                printProgressBar(loop, max_loop, t0)
            
            x = i / scale            
            y = j / scale
            z = k / scale

            density = trilinear_interpolation(self.density_accessor, x, y, z)
            velocity = trilinear_interpolation(self.velocity_accessor, x, y, z)
            turbulence_val = turbulence(x, y, z, i_min, i_max, noise_tile)
            # TODO: compute WL transform of energy, interpolate it, put it in there
            energy = 1

            velocity = velocity + 2**(-5/6) * energy * turbulence_val

            # print("%s %s %s: density: %s  velocity: %s" % (i, j, k, density, velocity))
            
            N_density_accessor.setValueOn((i, j, k), density)
            N_velocity_accessor.setValueOn((i, j, k), velocity)

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
smoke.make_higher_res(100, "test_enhanced.vdb")
