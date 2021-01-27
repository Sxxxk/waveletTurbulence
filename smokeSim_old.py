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



class FluidCube:
    def __init__(self):
      self.size = 40 # int
      self.dt = 0.1 # float
      self.diff = 0.001 # float
      self.visc = 0.001 # float

      self.s = None # pointeur vers float (tableau de dimension 3)
      self.density = None # pointeur vers float (tableau de dimension 3)

      self.Vx = None # pointeur vers float (tableau de dimension 3)
      self.Vy = None # pointeur vers float (tableau de dimension 3)
      self.Vz = None # pointeur vers float (tableau de dimension 3)

      self.Vx0 = None # pointeur vers float (tableau de dimension 3)
      self.Vy0 = None # pointeur vers float (tableau de dimension 3)
      self.Vz0 = None # pointeur vers float (tableau de dimension 3)

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
        self.diff = diffusion
        self.visc = viscosity

        self.s = np.zeros((size, size, size))
        self.density = np.zeros((size, size, size))

        self.Vx = np.zeros((size, size, size))
        self.Vy = np.zeros((size, size, size))
        self.Vz = np.zeros((size, size, size))
        
        self.Vx0 = np.zeros((size, size, size))
        self.Vy0 = np.zeros((size, size, size))
        self.Vz0 = np.zeros((size, size, size))
    
    def addDensity(self, x, y, z, amount):
        """Add density

        Args:
            x (int): 
            y (int): 
            z (int): 
            amount (float): 
        """
        self.density[x, y, z] += amount
    
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
        self.Vx[x, y, z] += amountX
        self.Vy[x, y, z] += amountY
        self.Vz[x, y, z] += amountZ

    def makeStep(self):
        diffuse(1, self.Vx0, self.Vx, self.visc, self.dt, 4, self.size)
        diffuse(2, self.Vy0, self.Vy, self.visc, self.dt, 4, self.size)
        diffuse(3, self.Vz0, self.Vz, self.visc, self.dt, 4, self.size)

        project(self.Vx0, self.Vy0, self.Vz0, self.Vx, self.Vy, 4, self.size)

        advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0, self.Vz0, self.dt, self.size)
        advect(2, self.Vy, self.Vy0, self.Vx0, self.Vy0, self.Vz0, self.dt, self.size)
        advect(3, self.Vz, self.Vz0, self.Vx0, self.Vy0, self.Vz0, self.dt, self.size)

        project(self.Vx, self.Vy, self.Vz, self.Vx0, self.Vy0, 4, self.size)
        
        diffuse(0, self.s, self.density, self.diff, self.dt, 4, self.size)
        advect(0, self.density, self.s, self.Vx, self.Vy, self.Vz, self.dt, self.size)
    
    def saveImg(self, filename):
        img = Image.new("RGB", (self.size, self.size))
        px = img.load()
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    # print(self.density[i, j, k])
                    px[i, j] = (int(px[i, j][0] + 50*self.density[i, j, k]), 0, 0)

        img.save("{}.png".format(filename))


def dist3d(a, b):
    i, j, k = a
    x, y, z = b
    return math.sqrt((i - x) ** 2 + (j - y) ** 2 + (k - z)**2)


if __name__ == "__main__":
    N = 40

    print("==== initializing fluid ====")
    fluid = FluidCube()
    fluid.create(N, 0.01, 0.1, 0.1)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if dist3d((i, j, k), (20, 10, 20)) < 2:
                    fluid.addDensity(i, j, k, 50)
                rd1 = np.random.rand(3)[0] - 0.5
                rd2 = np.random.rand(3)[1] - 0.5
                rd3 = np.random.rand(3)[2] - 0.5
                fluid.addVelocity(i, j, k, rd1, 10 + rd2, rd3)

    fluid.saveImg("img/img0")
    print("==== simulation ====")
    for step in range(200):
        print("step ", step)
        fluid.makeStep()
        fluid.saveImg("img/img{}".format(step + 1))