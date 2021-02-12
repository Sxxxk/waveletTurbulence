import pyopenvdb as vdb
from functools import reduce
import numpy as np


class Voxel:
    def __init__(self, coord=(0, 0, 0), dens_value=0, vel_value=(0, 0, 0)):
        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]
        self.dens_value = dens_value
        self.vel_value = vel_value


class Grid:
    def __init__(self):
        self.filename = 'fluid_data_0190.vdb'
        self.minBBox = (0, 0, 0)
        self.maxBBox = (0, 0, 0)
        self.grids = {}
        self.metadata = None
        self.densAccess = None
        self.velAccess = None
    
    def create(self, file='fluid_data_0190.vdb'):
        self.filename = file
        grids, self.metadata = vdb.readAll(file)
        for g in grids:
            self.grids[g.name] = g
            
        self.minBBox, self.maxBBox = self.grids['density'].evalActiveVoxelBoundingBox()
        print("minBBox : {} \t maxBBox : {}".format(self.minBBox, self.maxBBox))
        self.printGrids()
        self.densAccess = self.grids['density'].getConstAccessor()
        self.velAccess = self.grids['velocity'].getConstAccessor()
    
    
    def printGrids(self):
        for i, grid in enumerate(self.grids):
            print("{} : {} \t\t ({})".format(i, grid, str(type(self.grids[grid])).split(" ")[1][1:-2]))
    
    def _vel(self, value):
        return self.velAccess.getValue(value)
    
    def _dens(self, value):
        return self.densAccess.getValue(value)
    
    def size(self):
        x = self.maxBBox[0] - self.minBBox[0] + 1
        y = self.maxBBox[1] - self.minBBox[1] + 1
        z = self.maxBBox[2] - self.minBBox[2] + 1
        return (x, y, z)
    
    
    def toArray(self):
        size = self.size()
        array = np.zeros(size, dtype=Voxel)
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    dens_value = self._dens((x, y, z))
                    vel_value = self._vel((x, y, z))
                    vel = Voxel((x, y, z), dens_value, vel_value)
                    array[x, y, z] = vel
        
        return array


    def nbActiveVoxel(self):
        size = self.size()
        
        count = 0
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    if self.densAccess.isValueOn((x, y, z)):
                        count += 1
        
        return count
                    
        


if __name__ == "__main__":
    g = Grid()
    g.create()
    grid = g.toArray()