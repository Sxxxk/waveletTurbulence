import pywt
import numpy as np
import matplotlib.pyplot as plt
def haar_wavelet_3d(data):
    # 高通滤波
    aaa = (data[::2, ::2, ::2] + data[1::2, ::2, ::2] + data[::2, 1::2, ::2] + data[1::2, 1::2, ::2] +
           data[::2, ::2, 1::2] + data[1::2, ::2, 1::2] + data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2]) / 8.0

    aad = (data[::2, ::2, ::2] + data[1::2, ::2, ::2] + data[::2, 1::2, ::2] + data[1::2, 1::2, ::2] -
           (data[::2, ::2, 1::2] + data[1::2, ::2, 1::2] + data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2])) / 8.0

    ada = (data[::2, ::2, ::2] + data[1::2, ::2, ::2] - (data[::2, 1::2, ::2] + data[1::2, 1::2, ::2]) +
           data[::2, ::2, 1::2] + data[1::2, ::2, 1::2] - (data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2])) / 8.0

    daa = (data[::2, ::2, ::2] - (data[1::2, ::2, ::2] + data[::2, 1::2, ::2] + data[1::2, 1::2, ::2]) +
           data[::2, ::2, 1::2] - (data[1::2, ::2, 1::2] + data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2])) / 8.0

    # 低通滤波
    add = (data[::2, ::2, ::2] + data[1::2, ::2, ::2] + data[::2, 1::2, ::2] + data[1::2, 1::2, ::2] +
           data[::2, ::2, 1::2] + data[1::2, ::2, 1::2] + data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2]) / 8.0
    dad = (data[::2, ::2, 1::2] + data[1::2, ::2, 1::2] + data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2]) / 4.0
    dda = (data[1::2, ::2, ::2] + data[1::2, 1::2, ::2] + data[1::2, ::2, 1::2] + data[1::2, 1::2, 1::2]) / 4.0
    ddd = (data[::2, 1::2, ::2] + data[1::2, 1::2, ::2] + data[::2, 1::2, 1::2] + data[1::2, 1::2, 1::2]) / 4.0

    return aaa, aad, ada, daa, add, dad, dda, ddd

# 生成一个三维数据，这里假设是一个立方体
data = np.random.random((8, 8, 8))

# 进行三维离散小波变换
coeffs = haar_wavelet_3d(data)

# 可视化一个示例，这里只展示了 add 分量
import matplotlib.pyplot as plt
plt.imshow(coeffs[4][:, :, 0], cmap='gray')
plt.title('add Component')
plt.show()







# 进行三维离散小波变换
coeffs = pywt.dwtn(data, 'haar')  # 使用Haar小波

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
print(dad)
plt.imshow(dad[:, :, 0], cmap='gray')
plt.title("add")
plt.show()
