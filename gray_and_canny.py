import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


import cv2 as cv
import numpy as np

#全局阈值
def threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print("threshold value %s"%ret)
    cv.namedWindow("binary0", cv.WINDOW_NORMAL)
    cv.imshow("binary0", binary)
    cv.imwrite('F:\\TestCode\\pic\\binary0.jpg',binary)

#局部阈值
def local_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    binary =  cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 25, 10)
    cv.namedWindow("binary1", cv.WINDOW_NORMAL)
    cv.imshow("binary1", binary)
    cv.imwrite('F:\\TestCode\\pic\\binary1.jpg',binary)

#用户自己计算阈值
def custom_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    print("mean:",mean)
    ret, binary =  cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.namedWindow("binary2", cv.WINDOW_NORMAL)
    cv.imshow("binary2", binary)
    cv.imwrite('F:\\TestCode\\pic\\binary2.jpg',binary)

src = cv.imread('F:\\TestCode\pic\\92.jpg')
cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
threshold_demo(src)
local_threshold(src)
custom_threshold(src)
cv.waitKey(0)
cv.destroyAllWindows()

import cv2 as cv
import numpy as np
import os
path = r'F:\\ater_json\\moulde_img_origin_1-864_864\\img\\'
filenames = os.listdir(path)
c = 0
value = 3
for filename in filenames:
    print(filename)
    img = cv.imread(path+filename, 1)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 把输入图像灰度化
    value = 203
    print(value)
    cv.imwrite('F:\\ater_json\\moulde_img_origin_1-864_864\\delect\\'+str(value)+'_'+filename.split('.')[0]+'_gray_.jpg',img)
    # 自适应阈值化能够根据图像不同区域亮度分布，改变阈值
    img =  cv.adaptiveThreshold(img,
                                255,
                                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY,
                                value, 0)
    cv.imwrite('F:\\ater_json\\moulde_img_origin_1-864_864\\delect\\'+str(value)+'_'+filename, img)
    c = c+1
# img = cv2.imread("F:\TestCode\pic\92.jpg", 4)
# cv2.imshow("ori", img)
# cv2.imwrite("canny2.jpg", cv2.Canny(img, 100, 100))
# cv2.imshow("canny", cv2.imread("canny2.jpg"))
# cv2.waitKey()
# cv2.destroyAllWindows()



