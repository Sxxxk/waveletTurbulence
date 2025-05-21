import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte
import os
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
def compute_local_entropy_single(img_gray, radius=5):
    img_ubyte = img_as_ubyte(img_gray)
    entropy_map = entropy(img_ubyte, disk(radius))
    mean_entropy = np.mean(entropy_map)
    max_entropy = np.max(entropy_map)
    return mean_entropy, max_entropy, entropy_map

def compare_local_entropy(image_paths, radius=5):
    results = []

    for path in image_paths:
        # 读取灰度图
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[跳过] 图像无法读取: {path}")
            continue

        entropy_val,max_entropy_val,entropy_map = compute_local_entropy_single(img, radius)
        results.append({
            'path': path,
            'image': img,
            'entropy': entropy_val,
            #'entropy_max':max_entropy_val,
            'entropy_map': entropy_map
        })

    # 按熵值降序排序
    results.sort(key=lambda x: x['entropy'], reverse=True)

    # 打印结果
    print("\n图像局部熵对比结果（由高到低）:")
    for i, res in enumerate(results):
        filename = os.path.basename(res['path'])
        print(f"{i+1}. {filename}: 平均局部熵 = {res['entropy']:.4f}")
        #print(f"{i + 1}. {filename}: 最高局部熵 = {res['entropy_max']:.4f}")
    # 可视化
    cols = len(results)
    plt.figure(figsize=(4 * cols, 6))

    for i, res in enumerate(results):
        plt.subplot(2, cols, i + 1)
        plt.imshow(res['image'], cmap='gray')
        plt.title(f"{os.path.basename(res['path'])}", fontsize=20,fontweight='bold')
        plt.axis('off')

        plt.subplot(2, cols, i + 1 + cols)
        plt.imshow(res['entropy_map'], cmap='inferno')
        plt.title(f"平均熵: {res['entropy']:.4f}", fontsize=20,fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    return results
image_paths = [
    'DetailSmokePicture/default.png',
    'DetailSmokePicture/upSampling.png',
    'DetailSmokePicture/ours.png',
]

compare_local_entropy(image_paths, radius=5)