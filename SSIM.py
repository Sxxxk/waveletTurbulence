from PIL import Image
import numpy as np
from skimage.measure import compare_ssim


def compar_images():
    img1 = Image.open(r'图层 0.png')
    img2 = Image.open(r'图层1.png')
    img2 = img2.resize(img1.size)
    # 转为灰度图像
    gray1 = np.array(img1.convert("L"))
    gray2 = np.array(img2.convert("L"))
    # 计算SSIM
    ssim = compare_ssim(gray1, gray2, multichannel=True)
    # 打印SSIM
    print('SSIM:', ssim)
compar_images()