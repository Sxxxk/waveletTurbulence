#!/usr/bin/env python3
import pyopenvdb as vdb
import math
import pywt
import numpy as np
from itertools import product
from time import time

from interpolate import trilinear_interpolation, interpolate_wl
from noise import load_noise_tile, turbulence


class Smoke:
    def __init__(self, filename):
        self.dt = 0.1
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
            elif grid.name == 'velocity':
                #读取速度场
                self.velocity_grid = vdb.read(filename, grid.name)
                #获得速度场的访问器。getAccessor() 函数可以获取一个特定网格的访问器，然后使用该访问器来读取或修改网格中的数据。
                self.velocity_accessor = self.velocity_grid.getAccessor()
        #如果没有获得速度场则返回错误
        if (self.density_grid is None or self.velocity_grid is None):
            raise Exception("Given file misses density or velocity grid.")
        
        # We get the max active index: + 1 because we want to iterate until there
        # but python for loops stops at max - 1
        #                              + 1 because we also take the empty voxel just
        # after the last one.
        # Thus the +2

        #evalActiveVoxelBoundingBox()的作用是计算包围活跃（非空）体素的边界框。在中文中，可以将其解释为计算包围活跃体素的边界框的功能。
        bbox = self.density_grid.evalActiveVoxelBoundingBox()
        self.min_voxel = max(min(bbox[0]) - 1 - 10, 0)
        self.n = max(bbox[1]) + 2 + 10
        print("Base resolution : %s." % self.n)
        print("Grids opened.")

    #能量计算,其使用的公式是原论文中的e(x) = 1/2*(|ux|^2)
    #energies是一个三维矩阵
    def compute_energies(self):
        energies = np.zeros((self.n, self.n, self.n))

        for (x, y, z) in product(range(self.n), repeat=3):
            #得到xyz方向上的速度
            vx, vy, vz = self.velocity_accessor.probeValue((x, y, z))[0]
            energies[x, y, z] = 0.5 * (vx * vx + vy * vy + vz * vz)

        return energies

    #
    def wl_transform(self, energies):
        # /!\ Dimension changes !!! Haar downsamples by 2 !
        #pywt是一个python的小波库
        #其中Haar是小波类型，具体过程可见https://blog.csdn.net/qq_43665602/article/details/127176186。Haar小波的滤波会让分辨率降低到之前的1/2
        #pywt.dwtn的作用是对输入的n维数组进行n维离散小波变换
        #输入的二维数组，可以是图像数据或其他二维信号。
        #wavelet：用于变换的小波基函数，可以是字符串（表示内置的小波函数）或是小波函数对象。
        #返回值是一个Coeff字典。其中 key 指定 每个维度和值的转换类型都是 N 维 coefficients 数组。
        #具体解释和实例可查看https://pywavelets.readthedocs.io/en/latest/ref/nd-dwt-and-idwt.html#single-level-dwtn
        coeff_dict = pywt.dwtn(energies, 'Haar')
        return coeff_dict

    #平流操作
    def advect(self, d0, velocity_accessor, end, result_grid_accessor):
        """Advect density

        Args:
            d0 (numpy array): density before
        """
        N = end - 1
        dt0 = self.dt * end
        
        for (i, j, k) in product(range(1, N + 1), repeat=3):
            vel_value = velocity_accessor.getValue((i, j, k))
            x = i - dt0 * vel_value[0]
            y = j - dt0 * vel_value[1]
            z = k - dt0 * vel_value[2]
            
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
            u0 = 1 - u1

            value = s0 * (t0 * (u0 * d0[i0, j0, k0] + u1 * d0[i0, j0, k1])
                                       + t1 * (u0 * d0[i0, j1, k0] + u1 * d0[i0, j1, k1])) \
                                 + s1 * (t0 * (u0 * d0[i1, j0, k0] + u1 * d0[i1, j0, k1])
                                       + t1 * (u0 * d0[i1, j1, k0] + u1 * d0[i1, j1, k1]))

            # print(value, d0[i, j, k])
            result_grid_accessor.setValueOn((i, j, k), value)
        
    #上采样，其中N是上采样之后的分辨率
    def make_higher_res(self, N, filename):
        # Initializing output grids
        #初始化密度场以及名字和Accessor。密度场的元素为Float型
        N_density_grid = vdb.FloatGrid()
        N_density_grid.name = 'density'
        N_density_accessor = N_density_grid.getAccessor()
        #初始化上采样之后的速度场。速度场大的元素为Vec3型
        N_velocity_grid = vdb.Vec3SGrid()
        N_velocity_grid.name = 'velocity'
        N_velocity_accessor = N_velocity_grid.getAccessor()
        #返回一个三维数组，大小为N+1并全部初始化为0
        N_density_array = np.zeros((N+1, N+1, N+1))

        # 初始化必要的系数
        #scale是上采样的倍数
        scale = N / self.n
        min_voxel_enhanced = int(scale * self.min_voxel)

        #得到imin和imax，其具体计算可以参考博士论文。表示将湍流的能量注入到(n,N/2)这个区间内
        i_min = math.ceil(math.log(self.n, 2))#结果是以2为底，n 的对数，并向上取整
        i_max = math.floor(math.log(N / 2, 2))# 结果是以2为底 ，(N/2) 的对数，并向下取整

        #计算湍流的能量，即博士论文中的et(n/2).但是实际的计算并没有使用博士论文中的et，而是使用的原论文中的e(x) = 1/2*(|ux|^2)
        energies = self.compute_energies()
        #湍流能量的小波合成
        energies_wl_transform = self.wl_transform(energies)

        print("Loading noise tile.")
        noise_tile = load_noise_tile('noiseTile')
        print("Noise tile loaded.")

        print("Starting enhancing to new resolution %s." % N)
        #开始增强到新的分辨率
        # Utilitary variables used for the progress bar display
        #各项指标
        t0 = time()
        last_time = t0
        loop = 0
        max_loop = (N - min_voxel_enhanced)**3 - 1

        #上采样过程
        for (i, j, k) in product(range(min_voxel_enhanced, N), repeat=3):
            #输出过程信息
            if (time() - last_time > 0.5):
                printProgressBar(loop, max_loop, t0)
                last_time = time()
            
            x = i / scale            
            y = j / scale
            z = k / scale
            #对密度场直接进行三线性上采样
            density = trilinear_interpolation(self.density_accessor, x, y, z)
            #对速度场直接进行三线性上采样
            velocity = trilinear_interpolation(self.velocity_accessor, x, y, z)
            #得到湍流值
            turbulence_val = turbulence(x, y, z, i_min, i_max, noise_tile)

            energy_wl_transform_interpolated = interpolate_wl(energies_wl_transform, x, y, z)
            weight = 2**(-5/6) * energy_wl_transform_interpolated
            # if energy_wl_transform_interpolated != 0:
            #     print(i, j, k, energy_wl_transform_interpolated)

            velocity += weight * turbulence_val
            # if energy_wl_transform_interpolated != 0:
            #     print(i, j, k, weight * turbulence_val)

            # if (energy_wl_transform_interpolated != 0 and velocity[0] == 1 and velocity[1] == 0 and velocity[2] ==0):
            #     print("%s %s %s: density: %s  velocity: %s" % (i, j, k, density, velocity))
            
            N_density_array[i][j][k] = density
            N_velocity_accessor.setValueOn((i, j, k), velocity)

            loop += 1
        #根据新的速度场对密度场进行平流操作
        self.advect(N_density_array, N_velocity_accessor, N, N_density_accessor)

        print()
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


smoke = Smoke("fluid_data_0190.vdb")
smoke.make_higher_res(100, "result.vdb")
