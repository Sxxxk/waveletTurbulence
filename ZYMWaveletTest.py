import pywt
import numpy as np
import matplotlib.pyplot as plt

# from itertools import product
#
# # 定义范围
# min_voxel_enhanced = 0
# N = 3
# # 三重循环迭代
# for (i, j, k) in product(range(min_voxel_enhanced, N), repeat=3):
#     print(f"Processing voxel at ({i}, {j}, {k})")

#在GPU上可能需要一个[2，2，2]的线程组来实现
# 生成一个三维数据，这里假设是一个立方体
data = np.random.random((8, 8, 8))

def haar_wavelet_3d(data):
    # 高通滤波
    #以下为正确解值
    aaa = (data[::2, ::2, ::2] + data[1::2, ::2, ::2] + data[::2, 1::2, ::2] + data[1::2, 1::2, ::2] +
           data[::2, ::2, 1::2] + data[1::2, ::2, 1::2] + data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2]) / np.sqrt(8)

    aad = (data[::2, ::2, ::2] + data[1::2, ::2, ::2] + data[::2, 1::2, ::2] + data[1::2, 1::2, ::2] -
           (data[::2, ::2, 1::2] + data[1::2, ::2, 1::2] + data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2])) / np.sqrt(8)

    ada = (data[::2, ::2, ::2] + data[1::2, ::2, ::2] - (data[::2, 1::2, ::2] + data[1::2, 1::2, ::2]) +
           data[::2, ::2, 1::2] + data[1::2, ::2, 1::2] - (data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2])) / np.sqrt(8)

    daa = (data[::2, ::2, ::2] - data[1::2, ::2, ::2] + data[::2, 1::2, ::2] - data[1::2, 1::2, ::2] +
           data[::2, ::2, 1::2] - data[1::2, ::2, 1::2] + data[::2, 1::2, 1::2] - data[1::2, 1::2, 1::2]) / np.sqrt(8)

    # 低通滤波
    add =  (data[::2, ::2, ::2] + data[1::2, ::2, ::2] - data[::2, 1::2, ::2] - data[1::2, 1::2, ::2] -
            data[::2, ::2, 1::2] - data[1::2, ::2, 1::2] + data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2]) / np.sqrt(8)

    dad = (data[::2, ::2, ::2] - data[1::2, ::2, ::2] + data[::2, 1::2, ::2] - data[1::2, 1::2, ::2] -
           data[::2, ::2, 1::2] + data[1::2, ::2, 1::2] - data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2]) / np.sqrt(8)

    dda = (data[::2, ::2, ::2] - data[1::2, ::2, ::2] - data[::2, 1::2, ::2] + data[1::2, 1::2, ::2] +
           data[::2, ::2, 1::2] - data[1::2, ::2, 1::2] - data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2]) / np.sqrt(8)

    ddd =  (data[::2, ::2, ::2] - data[1::2, ::2, ::2] - data[::2, 1::2, ::2] + data[1::2, 1::2, ::2] -
           data[::2, ::2, 1::2] + data[1::2, ::2, 1::2] + data[::2, 1::2, 1::2] - data[1::2, 1::2, 1::2]) / np.sqrt(8)

    coeffs = {'aaa': aaa, 'aad': aad, 'ada': ada, 'daa': daa, 'add': add, 'dad': dad, 'dda': dda, 'ddd': ddd}
    return coeffs

# 生成一个三维数据，这里假设是一个立方体
data = np.random.random((8,8,8))
data1 = data

# 进行三维离散小波变换
coeffs = haar_wavelet_3d(data)
print(coeffs['aad'])
print("next")


# 可视化一个示例，这里只展示了 add 分量
import matplotlib.pyplot as plt
plt.imshow(coeffs['aad'][:, :, 0], cmap='gray')
plt.title('aad Component')
plt.show()


# 进行三维离散小波变换
# coeffs = haar_wavelet_3d(data)
#
# # 可视化一个示例，这里只展示了 add 分量
# import matplotlib.pyplot as plt
# plt.imshow(coeffs[4][:, :, 0], cmap='gray')
# plt.title('add Component')
# plt.show()







# 进行三维离散小波变换
coeffs = pywt.dwtn(data1, 'haar')  # 使用Haar小波

# 逐层展示高频和低频分量
# 高通滤波
aaa = coeffs['aaa']
aad = coeffs['aad']
ada = coeffs['ada']
daa = coeffs['daa']

# 低通滤波
add = coeffs['add']
dad = coeffs['dad']
dda = coeffs['dda']
ddd = coeffs['ddd']

# 在这里可以对高频和低频分量进行进一步处理，或者可视化

# 可视化一个示例，这里只展示了 aaa 分量
print(aad)
plt.imshow(aad[:, :, 0], cmap='gray')
plt.title("aad")
plt.show()



