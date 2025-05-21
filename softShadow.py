# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为黑体
# plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
# def compute_gradient_sharpness(image_path):
#     # 读取图像并转换为灰度
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError(f"无法加载图像：{image_path}")
#
#     # 可选：应用高斯模糊以减少噪声
#     img_blur = cv2.GaussianBlur(img, (3, 3), 0)
#
#     # 计算 Sobel 梯度（X 和 Y 方向）
#     grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
#
#     # 计算梯度幅值
#     grad_magnitude = cv2.magnitude(grad_x, grad_y)
#
#     # 计算平均梯度和最大梯度
#     mean_grad = np.mean(grad_magnitude)
#     max_grad = np.max(grad_magnitude)
#
#     return mean_grad, max_grad
#
# # 图像路径
# image_paths = {
#     # 'Hard': 'SoftShadowPicture/hard(1).png',
#     # 'PCF':  'SoftShadowPicture/PCF(1).png',
#     # 'Blur': 'SoftShadowPicture/blur(1).png',
#     # 'Ours': 'SoftShadowPicture/ours(1).png'
#     'Hard': 'SoftShadowPictureForest/hard.png',
#     'PCF':  'SoftShadowPictureForest/PCF.png',
#     'Blur': 'SoftShadowPictureForest/blur.png',
#     'Ours': 'SoftShadowPictureForest/ours.png'
#
# }
#
# # 存储结果
# mean_values = []
# max_values = []
#
# # 计算每张图像的梯度数据
# for name, path in image_paths.items():
#     mean, maxv = compute_gradient_sharpness(path)
#     mean_values.append(mean)
#     max_values.append(maxv)
#     print(f"{name} - 平均梯度: {mean:.2f}, 最大梯度: {maxv:.2f}")
#
# # 可视化：柱状图
# labels = list(image_paths.keys())
# x = np.arange(len(labels))  # 横坐标位置
# width = 0.35  # 每个柱子的宽度
#
# fig, ax = plt.subplots(figsize=(10, 6))
# bars1 = ax.bar(x - width/2, mean_values, width, label='平均梯度', color='skyblue')
# bars2 = ax.bar(x + width/2, max_values, width, label='最大梯度', color='orange')
#
# # 添加文字标签在柱子上方
# def add_value_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height:.2f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),  # 垂直方向偏移
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
# add_value_labels(bars1)
# add_value_labels(bars2)
#
# # 添加标签和图例
# ax.set_ylabel('梯度值')
# ax.set_title('不同方法的图像锐度对比（平均 & 最大梯度）')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
#
# plt.tight_layout()
# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

def compute_gradient_sharpness(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法加载图像：{image_path}")

    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)

    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    mean_grad = np.mean(grad_magnitude)

    return mean_grad

# 图像路径字典，两个场景
image_paths_geometry = {
    'Hard': 'SoftShadowPicture/hard(1).png',
    'PCF':  'SoftShadowPicture/PCF(1).png',
    'Blur': 'SoftShadowPicture/blur(1).png',
    'Ours': 'SoftShadowPicture/ours(1).png'
}

image_paths_forest = {
    'Hard': 'SoftShadowPictureForest/hard.png',
    'PCF':  'SoftShadowPictureForest/PCF.png',
    'Blur': 'SoftShadowPictureForest/blur.png',
    'Ours': 'SoftShadowPictureForest/ours.png'
}

labels = list(image_paths_geometry.keys())
x = np.arange(len(labels))  # [0, 1, 2, 3]
width = 0.35

# 存储两个场景的平均梯度
mean_geometry = []
mean_forest = []

for name in labels:
    mean_geom = compute_gradient_sharpness(image_paths_geometry[name])
    mean_for = compute_gradient_sharpness(image_paths_forest[name])
    mean_geometry.append(mean_geom)
    mean_forest.append(mean_for)
    print(f"{name} - 几何场景平均梯度: {mean_geom:.2f}, 森林场景平均梯度: {mean_for:.2f}")

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, mean_geometry, width, label='几何场景', color='skyblue')
bars2 = ax.bar(x + width/2, mean_forest, width, label='森林场景', color='orange')

# 添加文字标签在柱子上方
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_value_labels(bars1)
add_value_labels(bars2)

# 设置标签和图例
ax.set_ylabel('平均梯度值')
ax.set_title('不同方法在两种场景下的图像锐度对比（平均梯度）')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()
