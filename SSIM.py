from PIL import Image
import numpy as np
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr


def compar_images():
    img1 = Image.open(r'第三章实验\\128256128原始.png')
    img2 = Image.open(r'第三章实验\\128256128下采样.png')
    img2 = img2.resize(img1.size)
    # 转为灰度图像
    gray1 = np.array(img1.convert("L"))
    gray2 = np.array(img2.convert("L"))
    # 计算SSIM
    ssim = compare_ssim(gray1, gray2, multichannel=True)
    psnr = compare_psnr(gray1, gray2)
    # 打印SSIM
    print('SSIM:', ssim)
    print('PSNR:', psnr)
compar_images()