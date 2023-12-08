import numpy as np

def dwt_1d(signal):
    n = len(signal)
    h = [1, 1]  # 小波滤波器系数
    g = [1, -1]

    # 扩展信号长度为偶数
    if n % 2 != 0:
        signal = np.pad(signal, (0, 1), mode='constant', constant_values=0)

    # 低通滤波
    low_pass = np.convolve(signal, h, mode='valid')

    # 高通滤波
    high_pass = np.convolve(signal, g, mode='valid')

    # 下采样
    low_pass = low_pass[::2]
    high_pass = high_pass[::2]

    return low_pass, high_pass

def dwt_3d(data):
    # 对每个维度进行一维离散小波变换
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :], _ = dwt_1d(data[i, j, :])

    for i in range(data.shape[0]):
        for k in range(data.shape[2]):
            data[i, :, k], _ = dwt_1d(data[i, :, k])

    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            data[:, j, k], _ = dwt_1d(data[:, j, k])

    return data

# 测试
# 创建一个3x3x3的随机数组
data = np.random.random((3, 3, 3))

# 进行3D离散小波变换
result = dwt_3d(data)

print("原始数据:")
print(data)
print("\n变换后的数据:")
print(result)
