"""
Fluid Simulations in Python
Implementation of https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf

@author : Alexis
@date : jan. 2021
"""

import numpy as np
import math
from PIL import Image


def set_bnd(b, x, N):
    """set bound

    Args:
        b (int): 
        x (pointeur vers float):
        N (int):
    """
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            if b == 3:
                x[i, j, 0] = -x[i, j, 1  ]
                x[i, j, N-1] = -x[i, j, N-2]
            else:
                x[i, j, 0] = x[i, j, 1  ];
                x[i, j, N-1] = x[i, j, N-2];

    for k in range(1, N - 1):
        for i in range(1, N - 1):
            if b == 2:
                x[i, 0  , k] = -x[i, 1  , k]
                x[i, N-1, k] = -x[i, N-2, k]
            else:
                x[i, 0  , k] = x[i, 1  , k]
                x[i, N-1, k] = x[i, N-2, k]

    for k in range(1, N - 1):
        for j in range(1, N - 1):
            if b == 1:
                x[0  , j, k] = -x[1  , j, k]
                x[N-1, j, k] = -x[N-2, j, k]
            else:
                x[0  , j, k] = x[1  , j, k]
                x[N-1, j, k] = x[N-2, j, k]
    
    x[0, 0, 0] = 0.33 * (x[1, 0, 0] + x[0, 1, 0] + x[0, 0, 1])
    x[0, N-1, 0] = 0.33 * (x[1, N-1, 0] + x[0, N-2, 0] + x[0, N-1, 1])
    x[0, 0, N-1] = 0.33 * (x[1, 0, N-1] + x[0, 1, N-1] + x[0, 0, N - 2])
    x[0, N-1, N-1] = 0.33 * (x[1, N-1, N-1] + x[0, N-2, N-1] + x[0, N-1, N-2])
    x[N-1, 0, 0] = 0.33 * (x[N-2, 0, 0] + x[N-1, 1, 0] + x[N-1, 0, 1])
    x[N-1, N-1, 0] = 0.33 * (x[N-2, N-1, 0] + x[N-1, N-2, 0] + x[N-1, N-1, 1])
    x[N-1, 0, N-1] = 0.33 * (x[N-2, 0, N-1] + x[N-1, 1, N-1] + x[N-1, 0, N-2])
    x[N-1, N-1, N-1] = 0.33 * (x[N-2, N-1, N-1] + x[N-1, N-2, N-1] + x[N-1, N-1, N-2])


def lin_solve(b, x, x0, a, c, iterr, N):
    """lin solve

    Args:
        b (int):
        x (pointeur vers float):
        x0 (float):
        a (float):
        c (float):
        iterr (int): 
        N (int): 
    """
    cRecip = 1.0 / c;
    for k in range(0, iterr):
        for m in range(1, N - 1):
            for j in range(1, N - 1):
                for i in range(1, N - 1):
                    x[i, j, m] = (x0[i, j, m] 
                                + a*(x[i+1, j  , m  ]
                                    +x[i-1, j  , m  ]
                                    +x[i  , j+1, m  ]
                                    +x[i  , j-1, m  ]
                                    +x[i  , j  , m+1]
                                    +x[i  , j  , m-1]
                                )) * cRecip
        set_bnd(b, x, N)


def diffuse (b, x, x0, diff, dt, iterr, N):
    """diffuse

    Args:
        b (int):
        x (float*): 
        x0 (float*): 
        diff (float): 
        dt (float): 
        iterr (int): 
        N (int): 
    """
    a = dt * diff * (N - 2) * (N - 2)
    lin_solve(b, x, x0, a, 1 + 6 * a, iterr, N)


def project(velocX, velocY, velocZ, p, div, iterr, N):
    """project

    Args:
        velocX (float*): 
        velocY (float*): 
        velocZ (float*): 
        p (float*): 
        div (float*): 
        iterr (int): 
        N (int): 
    """
    for k in range(1, N - 1):
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                div[i, j, k] = -0.5 * (
                         velocX[i+1, j  , k  ]
                        -velocX[i-1, j  , k  ]
                        +velocY[i  , j+1, k  ]
                        -velocY[i  , j-1, k  ]
                        +velocZ[i  , j  , k+1]
                        -velocZ[i  , j  , k-1]
                    ) / N
                p[i, j, k] = 0

    set_bnd(0, div, N)
    set_bnd(0, p, N)
    lin_solve(0, p, div, 1, 6, iterr, N)
    
    for k in range(1, N - 1):
        for j in range(1, N - 1):
            for i in range(N - 1):
                velocX[i, j, k] -= 0.5 * ( p[i+1, j, k] - p[i-1, j, k] ) * N
                velocY[i, j, k] -= 0.5 * ( p[i, j+1, k] - p[i, j-1, k] ) * N
                velocZ[i, j, k] -= 0.5 * ( p[i, j, k+1] - p[i, j, k-1] ) * N

    set_bnd(1, velocX, N)
    set_bnd(2, velocY, N)
    set_bnd(3, velocZ, N)


def advect(b, d, d0, velocX, velocY, velocZ, dt, N):
    """advect

    Args:
        b (int): 
        d (float*): 
        d0 (float*): 
        velocX (float*): 
        velocY (float*): 
        velocZ (float*): 
        dt (float): 
        N (int): 
    """
    # float i0, i1, j0, j1, k0, k1;
    
    dtx = dt * (N - 2)
    dty = dt * (N - 2)
    dtz = dt * (N - 2)
    
    # float s0, s1, t0, t1, u0, u1;
    # float tmp1, tmp2, tmp3, x, y, z;
    
    # float ifloat, jfloat, kfloat;
    # int i, j, k;
    
    k = 1
    while k < N - 1:
        j = 1
        while j < N - 1:
            i = 1
            while i < N - 1:
                tmp1 = dtx * velocX[i, j, k]
                tmp2 = dty * velocY[i, j, k]
                tmp3 = dtz * velocZ[i, j, k]
                x    = i - tmp1 
                y    = j - tmp2
                z    = k - tmp3
                
                if x < 0.5:
                    x = 0.5
                if x > N + 0.5:
                    x = N + 0.5

                i0 = math.floor(x) 
                i1 = i0 + 1.0
                
                if y < 0.5:
                    y = 0.5 
                if y > N + 0.5:
                    y = N + 0.5
                
                j0 = math.floor(y)
                j1 = j0 + 1.0

                if z < 0.5:
                    z = 0.5
                if z > N + 0.5:
                    z = N + 0.5

                k0 = math.floor(z)
                k1 = k0 + 1.0
                
                s1 = x - i0 
                s0 = 1.0 - s1 
                t1 = y - j0
                t0 = 1.0 - t1
                u1 = z - k0
                u0 = 1.0 - u1
                
                i0i = int(i0)
                i1i = int(i1)
                j0i = int(j0)
                j1i = int(j1)
                k0i = int(k0)
                k1i = int(k1)
                
                d[i, j, k] = s0 * ( t0 * (u0 * d0[i0i, j0i, k0i] + u1 * d0[i0i, j0i, k1i])
                                  + t1 * (u0 * d0[i0i, j1i, k0i] + u1 * d0[i0i, j1i, k1i])) \
                           + s1 * ( t0 * (u0 * d0[i1i, j0i, k0i] + u1 * d0[i1i, j0i, k1i])
                                  + t1 * (u0 * d0[i1i, j1i, k0i] + u1 * d0[i1i, j1i, k1i]))
                
                i += 1
            j += 1
        k += 1

    set_bnd(b, d, N)



class Voxel:
    def __init__(self):
      self.density = 0
      self.Vx = 0
      self.Vy = 0
      self.Vz = 0


class FluidCube:
    def __init__(self):
        self.N = 40
        self.size = 42*42 # int
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

        self.u = np.zeros(size)
        self.u_prev = np.zeros(size)
        self.v = np.zeros(size)
        self.v_prev = np.zeros(size)
        self.w = np.zeros(size)
        self.w_prev = np.zeros(size)

        self.dens = np.zeros(size)
        self.dens_prev = np.zeros(size)
    

    def add_source(self, x, amount):
        """Add density

        Args:
            amount (array): 
        """
        for i in range(size):
            self.x[i] += self.dt * amount[i]


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

            set_bnd (N, b, x)
    
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
        
        self.set_bnd(N, b, d)


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
        
        self.project(u, v, w, u0, v0, w0)

        u, u0, v, v0, w, w0 = u0, u, v0, v, w0, w # SWAP

        self.advect(1, u, u0, u0, v0, w0)
        self.advect(2, v, v0, u0, v0, w0)
        self.advect(3, w, w0, u0, v0, w0)

        self.project(u, v, w, u0, v0, w0)


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

        h = 1/N
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
                    u[self.IX(i, j, k)]

    
    def saveImg(self, filename):
        img = Image.new("RGB", (self.size, self.size))
        px = img.load()
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    # print(self.density[i, j, k])
                    px[i, j] = (int(px[i, j][0] + 200*self.density[i, j, k]), 0, 0)

        img.save("{}.png".format(filename))


def dist3d(a, b):
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
                if dist3d((i, j, k), (10, 10, 20)) < 5:
                    fluid.addDensity(i, j, k, 1)
                    fluid.addVelocity(i, j, k, 5, 10, 5)
    fluid.saveImg("img/img0")
    print("==== simulation ====")
    for step in range(20):
        print("step ", step)
        fluid.makeStep()
        fluid.saveImg("img/img{}".format(step + 1))