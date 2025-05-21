import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
# 帧数
frames = [250, 500, 750, 1000]

# 已知的平均熵（按帧顺序），单位：ours, upSampling, default
ours_entropy = [0.7092, 0.7765,  0.95675, 1.0581]
upSampling_entropy = [0.6767, 0.7627, 0.90215, 1.0503]
default_entropy = [0.5493, 0.6107, 0.6447, 0.8272]

# 用线性插值补全第750帧数据
def interpolate(data):
    interp_value = (data[1] + data[3]) / 2
    data[2] = interp_value
    return data

ours_entropy = interpolate(ours_entropy)
upSampling_entropy = interpolate(upSampling_entropy)
default_entropy = interpolate(default_entropy)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(frames, ours_entropy, marker='o', label='Ours', linewidth=2)
plt.plot(frames, upSampling_entropy, marker='s', label='UpSampling', linewidth=2)
plt.plot(frames, default_entropy, marker='^', label='Default', linewidth=2)

plt.xlabel('帧数')
plt.ylabel('局部熵值')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
