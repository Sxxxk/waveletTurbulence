import pyopenvdb as vdb
from functools import reduce
import numpy as np


class Voxel:
    """
        Voxel object
        Store :
            - the coordinates (or rather indices) of the voxel in the parent grid
            - the density value of the voxel : float
            - the velocity value of the voxel : tuple (ux, uy, uz)
    """
    def __init__(self, coord=(0, 0, 0), dens_value=0, vel_value=(0, 0, 0)):
        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]
        self.dens_value = dens_value
        self.vel_value = vel_value


class Grid:
    """
        Store the grids and provide methods to extract data from the 
        density and velocity grids
    """
    def __init__(self):
        self.filename = 'fluid_data_0190.vdb'
        self.minBBox = (0, 0, 0)
        self.maxBBox = (0, 0, 0)
        self.grids = {}
        self.metadata = None
        self.densAccess = None
        self.velAccess = None
    
    def create(self, file='fluid_data_0190.vdb'):
        """Store the grids of a file in this object

        Args:
            file (str, optional): the path+name of the VDB file.
        """
        self.filename = file
        
        # Extract and store the grids
        grids, self.metadata = vdb.readAll(file)
        for g in grids:
            self.grids[g.name] = g
        
        # Voxel size
        self.voxelSize = float(self.grids['density'].info().split('\n')[15].split(" ")[-1])
        # Bounding box
        self.minBBox, self.maxBBox = self.grids['density'].evalActiveVoxelBoundingBox()
        print("minBBox : {} \t maxBBox : {}".format(self.minBBox, self.maxBBox))
        # Display grid names
        self.printGrids()
        
        # Create accessors
        self.densAccess = self.grids['density'].getConstAccessor()
        self.velAccess = self.grids['velocity'].getConstAccessor()
    
    
    def printGrids(self):
        """Display the grids
        """
        for i, grid in enumerate(self.grids):
            print("{} : {} \t\t ({})".format(i, grid, str(type(self.grids[grid])).split(" ")[1][1:-2]))
    
    def _vel(self, value):
        """Provide accessor for the velocity grid

        Args:
            value (tuple of 3 integer values): the indices to get the values : (x, y, z)

        Returns:
            tuple of 3 values: tuple with the velocity values : (ux, uy, uz)
        """
        return self.velAccess.getValue(value)
    
    def _dens(self, value):
        """Provide accessor for the density grid

        Args:
            value (tuple of 3 integer values): the indices to get the values : (x, y, z)

        Returns:
            float: density value
        """
        return self.densAccess.getValue(value)
    
    def size(self):
        """Compute the size of the density grid
        Using the difference between the min and max bounding box

        Returns:
            tuple of 3 values: size of the grid (x, y, z)
        """
        x = self.maxBBox[0] - self.minBBox[0] + 1
        y = self.maxBBox[1] - self.minBBox[1] + 1
        z = self.maxBBox[2] - self.minBBox[2] + 1
        return (x, y, z)
    
    
    def toArray(self):
        """Create a numpy array from the grids

        Returns:
            numpy.array: 3D array with Voxel objects as values
        """
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
        """Get the number of active voxels

        Returns:
            int: number of active voxels
        """
        size = self.size()
        
        count = 0
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    if self.densAccess.isValueOn((x, y, z)):
                        count += 1
        
        return count
                    
        

if __name__ == "__main__":
    file = 'fluid_data_0190.vdb'
    g = Grid()
    g.create(file)
    grid = g.toArray()
