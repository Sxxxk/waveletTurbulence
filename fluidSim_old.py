import numpy as np
import math
from PIL import Image


class Voxel:
    def __init__(self):
        self.Vx = 0
        self.Vy = 0
        self.Vz = 0
        self.density = 0


class FluidCube:
    def __init__(self):
      self.size = 40 # int
      self.dt = 0.1 # float
    
      self.voxels = None


    def create(self, size, diffusion, viscosity, dt):
        """Crée le fluide

        Args:
            size (int): 
            diffusion (int): 
            viscosity (int): 
            dt (float): 
        """
        self.size = size
        self.dt = dt

        self.voxels = np.zeros((size, size, size), dtype=Voxel)

    
    def addDensity(self, x, y, z, amount):
        """Add density

        Args:
            x (int): 
            y (int): 
            z (int): 
            amount (float): 
        """
        self.voxels[x, y, z].density += amount
    
    def addVelocity(self, x, y, z, amountX, amountY, amountZ):
        """Add velocity

        Args:
            x (int): 
            y (int): 
            z (int): 
            amountX (float): 
            amountY (float): 
            amountZ (float): 
        """
        self.voxels[x, y, z].Vx += amountX
        self.voxels[x, y, z].Vy += amountY
        self.voxels[x, y, z].Vz += amountZ

    def makeStep(self):
        # TODO
        return
    
    def saveImg(self, filename):
        img = Image.new("RGB", (self.size, self.size))
        px = img.load()
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    # print(self.density[i, j, k])
                    px[i, j] = (int(px[i, j][0] + 200*self.voxels[i, j, k].density), 0, 0)

        img.save("{}.png".format(filename))


def dist(a, b):
    i, j, k = a
    x, y, z = b
    return math.sqrt((i - x) ** 2 + (j - y) ** 2 + (k - z)**2)


if __name__ == "__main__":
    N = 40

    print("==== initializing fluid ====")
    fluid = FluidCube()
    fluid.create(N, 0.1, 0.1, 0.1)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if dist((i, j, k), (10, 10, 20)) < 5:
                    fluid.addDensity(i, j, k, 10)
                fluid.addVelocity(i, j, k, 5, 10, 5)
    fluid.saveImg("img/img0")
    print("==== simulation ====")
    for step in range(20):
        print("step ", step)
        fluid.makeStep()
        fluid.saveImg("img/img{}".format(step + 1))