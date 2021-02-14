"""
Fluid Simulations in Python
Implementation of https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf

@author : Alexis
@date : jan. 2021
"""

import numpy as np
import math
from PIL import Image
import openvdb as vdb


# Ne sert à rien
class Voxel:
    def __init__(self):
      self.density = 0
      self.Vx = 0
      self.Vy = 0
      self.Vz = 0


class FluidCube:
    def __init__(self):
        self.N = 40
        self.size = 42*42*42 # int
        self.dt = 0.1 # float
        self.diff = 0.001 # float
        self.visc = 0.001 # float

        self.u = None
        self.u_prev = None
        self.v = None
        self.v_prev = None
        self.w = None
        self.w_prev = None

        self.dens = None
        self.dens_prev = None

        self.frame = 0


    def IX(self, i, j, k):
        return (k * (N + 2) * (N + 2) + j * (N + 2) + i)


    def create(self, N, diffusion, viscosity, dt):
        """Crée le fluide

        Args:
            size (int): 
            diffusion (int): 
            viscosity (int): 
            dt (float): 
        """
        self.N = N
        self.size = (N+2) * (N+2) * (N+2)
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity

        self.u = np.zeros(self.size)
        self.u_prev = np.zeros(self.size)
        self.v = np.zeros(self.size)
        self.v_prev = np.zeros(self.size)
        self.w = np.zeros(self.size)
        self.w_prev = np.zeros(self.size)

        self.dens = np.zeros(self.size)
        self.dens_prev = np.zeros(self.size)
    

    def addDensity(self, i, j, k, value):
        self.dens_prev[self.IX(i, j, k)] += value
        self.dens[self.IX(i, j, k)] += value
    

    def addVelocity(self, i, j, k, vx, vy, vz):
        self.u_prev[self.IX(i, j, k)] += vx
        self.v_prev[self.IX(i, j, k)] += vy
        self.w_prev[self.IX(i, j, k)] += vz


    def add_source(self, x, amount):
        """Add density

        Args:
            amount (array): 
        """
        for i in range(self.size):
            x[i] += self.dt * amount[i]


    def diffuse(self, b, x, x0):
        """diffuse

        Args:
            b (int): 
            x (array): 
            x0 (array): 
        """
        N = self.N
        dt = self.dt
        diff = self.diff

        a= dt * diff * N * N * N # Quid du dernier N ?
        for k in range(20):
            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    for k in range(1, N + 1):
                        x[self.IX(i, j, k)] = ( x0[self.IX(i, j, k)] + a * (
                              x0[self.IX(i-1, j, k)] + x0[self.IX(i+1, j, k)]
                            + x0[self.IX(i, j-1, k)] + x0[self.IX(i, j+1, k)]
                            + x0[self.IX(i, j, k-1)] + x0[self.IX(i, j, k+1)]
                            - 6 * x0[self.IX(i, j, k)])) / (1 + 6 * a)

            self.set_bnd(b, x)
    
    def advect(self, b, d, d0, u, v, w):
        """advect

        Args:
            b (int): 
            d (array): 
            d0 (array): 
            u (array): 
            v (array): 
            w (array): 
        """
        N = self.N

        dt0 = self.dt * N
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                for k in range(1, N + 1):
                    x = i - dt0 * u[self.IX(i, j, k)]
                    y = j - dt0 * v[self.IX(i, j, k)]
                    z = k - dt0 * w[self.IX(i, j, k)]
                    
                    if x < 0.5:
                        x = 0.5
                    if x > N + 0.5:
                        x = N + 0.5
                    i0 = int(x)
                    i1 = i0 + 1

                    if y < 0.5:
                        y = 0.5
                    if y > N + 0.5:
                        y = N + 0.5
                    j0 = int(y)
                    j1 = j0 + 1

                    if z < 0.5:
                        z = 0.5
                    if z > N + 0.5:
                        z = N + 0.5
                    k0 = int(z)
                    k1 = k0 + 1

                    s1 = x - i0
                    s0 = 1 - s1
                    t1 = y - j0
                    t0 = 1 - t1
                    u1 = z - k0
                    u0 = 1 - k1

                    d[self.IX(i, j, k)] = s0 * (t0 * (u0 * d0[self.IX(i0, j0, k0)] + u1 * d0[self.IX(i0, j0, k1)])
                                              + t1 * (u0 * d0[self.IX(i0, j1, k0)] + u1 * d0[self.IX(i0, j1, k1)])) \
                                        + s1 * (t0 * (u0 * d0[self.IX(i1, j0, k0)] + u1 * d0[self.IX(i1, j0, k1)])
                                              + t1 * (u0 * d0[self.IX(i1, j1, k0)] + u1 * d0[self.IX(i1, j1, k1)]))
        
        self.set_bnd(b, d)


    def dens_step(self, x, x0, u, v, w):
        """density step

        Args:
            x (array): 
            x0 (array): 
            u (array): 
            v (array): 
            w (array): 
        """
        N = self.N
        diff = self.diff
        dt = self.dt

        self.add_source(x, x0)
        x, x0 = x0, x # Swap
        self.diffuse(0, x, x0)

        x, x0 = x0, x # Swap
        self.advect(0, x, x0, u, v, w)

    
    def vel_step(self, u, v, w, u0, v0, w0):
        """velocity step

        Args:
            u (array): 
            v (array): 
            w (array): 
            u0 (array): 
            v0 (array): 
            w0 (array): 
        """
        N = self.N
        visc = self.visc
        dt = self.dt

        self.add_source(u, u0)
        self.add_source(v, v0)
        self.add_source(w, w0)

        u, u0 = u0, u # SWAP
        self.diffuse(1, u, u0)
        v, v0 = v0, v # SWAP
        self.diffuse(1, v, v0)
        w, w0 = w0, w # SWAP
        self.diffuse(1, w, w0)
        
        self.project(u, v, w, u0, v0) # u0, v0 ne servent pas spécifiquement en réalité

        u, u0, v, v0, w, w0 = u0, u, v0, v, w0, w # SWAP

        self.advect(1, u, u0, u0, v0, w0)
        self.advect(2, v, v0, u0, v0, w0)
        self.advect(3, w, w0, u0, v0, w0)

        self.project(u, v, w, u0, v0)


    def project(self, u, v, w, p, div):
        """project

        Args:
            u (array): 
            v (array): 
            w (array): 
            p (array): 
            div (array): 
        """
        N = self.N

        h = 1.0 / N

        for i in range(1, N + 1):
            for j in range(1, N + 1):
                for k in range(1, N + 1):
                    div[self.IX(i, j, k)] = -0.5 * h * (
                            u[self.IX(i+1, j  , k  )] - u[self.IX(i-1, j  , k  )]
                          + v[self.IX(i  , j+1, k  )] - u[self.IX(i  , j-1, k  )]
                          + w[self.IX(i  , j  , k+1)] - u[self.IX(i  , j  , k-1)]
                    )
                    p[self.IX(i, j, k)] = 0
        
        self.set_bnd(0, div)
        self.set_bnd(0, p)

        for n_k in range(20):
            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    for k in range(1, N + 1):
                        p[self.IX(i, j, k)] = (div[self.IX(i, j, k)] 
                            + p[self.IX(i-1, j  , k  )] + p[self.IX(i+1, j  , k  )]
                            + p[self.IX(i  , j-1, k  )] + p[self.IX(i  , j+1, k  )]
                            + p[self.IX(i  , j  , k-1)] + p[self.IX(i  , j  , k+1)]
                        ) / 6
            self.set_bnd(0, p)

        for i in range(1, N + 1):
            for j in range(1, N + 1):
                for k in range(1, N + 1):
                    u[self.IX(i, j, k)] -= 0.5 * (p[self.IX(i+1, j  , k  )] - p[self.IX(i-1, j  , k  )]) / h
                    v[self.IX(i, j, k)] -= 0.5 * (p[self.IX(i  , j+1, k  )] - p[self.IX(i  , j-1, k  )]) / h
                    w[self.IX(i, j, k)] -= 0.5 * (p[self.IX(i  , j  , k+1)] - p[self.IX(i  , j  , k-1)]) / h
        
        self.set_bnd(1, u)
        self.set_bnd(2, v)
        self.set_bnd(3, w)

    
    def set_bnd(self, b, x):
        """set boundaries

        Args:
            b (int): 
            x (array): 
        """
        for i in range(1, N + 1):
            if b == 1:
                x[self.IX(0  , i  , i  )] = - x[self.IX(1  , i  , i  )]
                x[self.IX(N+1, i  , i  )] = - x[self.IX(N  , i  , i  )]
            else:
                x[self.IX(0  , i  , i  )] =   x[self.IX(1  , i  , i  )]
                x[self.IX(N+1, i  , i  )] =   x[self.IX(N  , i  , i  )]
            
            if b == 2:
                x[self.IX(i  , 0  , i  )] = - x[self.IX(i  , 1  , i  )]
                x[self.IX(i  , N+1, i  )] = - x[self.IX(i  , N  , i  )]
            else:
                x[self.IX(i  , 0  , i  )] =   x[self.IX(i  , 1  , i  )]
                x[self.IX(i  , N+1, i  )] =   x[self.IX(i  , N  , i  )]
            
            if b == 3:
                x[self.IX(i  , i  , 0  )] = - x[self.IX(i  , i  , 1  )]
                x[self.IX(i  , i  , N+1)] = - x[self.IX(i  , i  , N  )]
            else:
                x[self.IX(i  , i  , 0  )] =   x[self.IX(i  , i  , 1  )]
                x[self.IX(i  , i  , N+1)] =   x[self.IX(i  , i  , N  )]
        
        x[self.IX(0  , 0  , 0  )] = 0.33 * (x[self.IX(1  , 0  , 0  )] + x[self.IX(0  , 1  , 0  )] + x[self.IX(0  , 0  , 1  )])
        x[self.IX(N+1, 0  , 0  )] = 0.33 * (x[self.IX(N  , 0  , 0  )] + x[self.IX(N+1, 1  , 0  )] + x[self.IX(N+1, 0  , 1  )])
        x[self.IX(0  , N+1, 0  )] = 0.33 * (x[self.IX(1  , N  , 0  )] + x[self.IX(0  , N+1, 0  )] + x[self.IX(0  , N+1, 1  )])
        x[self.IX(0  , 0  , N+1)] = 0.33 * (x[self.IX(1  , 0  , N+1)] + x[self.IX(0  , 1  , N+1)] + x[self.IX(0  , 0  , N  )])
        x[self.IX(N+1, N+1, 0  )] = 0.33 * (x[self.IX(N  , N+1, 0  )] + x[self.IX(N+1, N  , 0  )] + x[self.IX(N+1, N+1, 1  )])
        x[self.IX(0  , N+1, N+1)] = 0.33 * (x[self.IX(1  , N+1, N+1)] + x[self.IX(0  , N  , N+1)] + x[self.IX(0  , N+1, N  )])
        x[self.IX(N+1, 0  , N+1)] = 0.33 * (x[self.IX(N  , 0  , N+1)] + x[self.IX(N+1, 1  , N+1)] + x[self.IX(N+1, 0  , N  )])
        x[self.IX(N+1, N+1, N+1)] = 0.33 * (x[self.IX(N  , N+1, N+1)] + x[self.IX(N+1, N  , N+1)] + x[self.IX(N+1, N+1, N  )])

    
    def make_step(self):
        # Get density and velocity values from UI (or before)
        if self.frame > 0:
            self.u_prev = np.copy(self.u)
            
        # velocity step
        self.vel_step(self.u, self.v, self.w, self.u_prev, self.v_prev, self.w_prev)
        # density step
        self.dens_step(self.dens, self.dens_prev, self.u, self.v, self.w)


    def saveImg(self, filename):
        N = self.N
        img = Image.new("RGB", (N, N))
        px = img.load()
        for i in range(1, N+1):
            for j in range(1, N+1):
                for k in range(1, N+1):
                    # print(self.density[i, j, k])
                    # print(i, j, k)
                    try:
                        px[i-1, j-1] = (int(px[i-1, j-1][0] + self.dens[self.IX(i, j, k)]), 0, 0)
                    except:
                        px[i-1, j-1] = 10

        img.save("{}.png".format(filename))
    

    def saveVDB(self, filename):
        density_grid = vdb.FloatGrid()
        density_grid.name = "density"
        dens_accessor = density_grid.getAccessor()
        vel_grid = vdb.Vec3SGrid()
        vel_grid.name = "v"
        vel_accessor = vel_grid.getAccessor()
        for i in range(1, N+1):
            for j in range(1, N+1):
                for k in range(1, N+1):
                    dens_value = self.dens[self.IX(i, j, k)]
                    dens_accessor.setValueOn((i, j, k), dens_value)
                    vel_value = np.array([self.u[self.IX(i, j, k)],
                                        + self.v[self.IX(i, j, k)],
                                        + self.w[self.IX(i, j, k)]])
                    vel_accessor.setValueOn((i, j, k), vel_value)
        vdb.write("{}.vdb".format(filename), grids = [density_grid, vel_grid])


def dist3d(a, b):
    i, j, k = a
    x, y, z = b
    return math.sqrt((i - x) ** 2 + (j - y) ** 2 + (k - z)**2)

def dist_ax(a, ax):
    i, j, k = a
    x, y, z = ax
    return math.sqrt((i - x) ** 2 + (k - z)**2)



if __name__ == "__main__":
    N = 40

    print("==== initializing fluid ====")
    fluid = FluidCube()
    fluid.create(N, 0.001, 0.001, 0.1)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # Density
                if dist3d((i, j, k), (20, 5, 20)) < 4:
                    rd = np.random.rand(1)[0] - 0.5
                    fluid.addDensity(i, j, k, 10 + 5 * rd)
                # Velocity
                if dist_ax((i, j, k), (20, 0, 20)) <= 1.5:
                    rd1 = np.random.rand(3)[0] - 0.5
                    rd2 = np.random.rand(3)[1] - 0.5
                    rd3 = np.random.rand(3)[2] - 0.5
                    fluid.addVelocity(i, j, k, rd1, 5 + rd2, rd3)
    fluid.saveImg("img/img0")
    print("==== simulation ====")
    for step in range(1, 50):
        print("step ", step)
        fluid.make_step()
        fluid.saveImg("img/img{}".format(step))
        fluid.saveVDB("vdb/fluid{}".format(step))