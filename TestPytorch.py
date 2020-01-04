#coding=utf-8
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.variable as Variable
from train_process_visualize import acc_visualize

# acc_visualize('acc', 1, 1)
for i in range(0,10):
    acc_visualize('xxx/2', i, i, 1, False)
    print(i)
acc_visualize('xxx/2', i, i, 1, True)

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.con1 = nn.Conv2d(3, 6, 5, 1, 2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.con2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120,84)
        self.fc_relu2 = nn.ReLU()
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = self.con1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.con2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# def convolution_layer(x, kernal_size = 3, stride = 1, padding = 0, num = 1):
#
# def relu_layer(x):
#
# def pooling(x, kernal_size = 3, stride = 1, padding = 0):
#
# def fc_layer(x, output_num = 2):



# import numpy as np
# from scipy import optimize
# import math
#
# def f_1(x, A, B):
#     return A * x + B
#
# def linear_regression(x, y):
#   N = len(x)
#   sumx = sum(x)
#   sumy = sum(y)
#   sumx2 = sum(x ** 2)
#   sumxy = sum(x * y)
#   A = np.mat([[N, sumx], [sumx, sumx2]])
#   b = np.array([sumy, sumxy])
#   return np.linalg.solve(A, b)
#
# list = np.array([[100.0,28.0],[97.0,23.5],[83.5,20.5],[71.5,16.5],[57.5,13.5],[44.5,10.5],[31.5,7.5],[15.0,7.0]])
#
# # list_x = np.array([100.0,97.0,83.5,71.5,57.5,44.5,31.5,15.0])
# # list_y = np.array([28.0,23.5,20.5,16.5,13.5,10.5,7.5,7.0])
#
# # list_x = np.array([5.5,4,4,4,5.5])
# # list_y = np.array([1,2,3,4,5])
#
# # list_x = np.array([102.5,61.0,47.0,98.0,33.5,84.5,20.0,70.0,6.0,57.0,43.0,31.0,13.0])
# # list_y = np.array([27.5,28.0,25.0,23.5,22.0,20.0,18.5,16.5,15.5,13.5,10.5,7.5,7.0])
#
# list_x = np.array([102.0,97.0,124.5,83.5,69.5,120.0,56.0,106.5,42.0,93.0,28.0,13.0,79.5,66.0])
# list_y = np.array([27.5,23.0,19.5,20.0,17.0,15.5,14.0,12.0,10.5,9.0,7.5,7.0,5.5,2.0])
# import matplotlib.pyplot as plt
# print(type(list_y))
# xx =[]
# yy =[]
# for i in range(len(list_y)):
#   # plt.scatter(list_x[:], list_y[:], 3, "red")
#   xx.append(list_x[i])
#   yy.append(list_y[i])
#   print(list_x[i],list_y[i])
#   plt.scatter(xx[:],yy[:],3,'red')
#   plt.xlabel('x')
#   plt.ylabel('y')
#   plt.show()
#
# X1 = list_x
# Y1 = list_y
#
# a0, a1 = linear_regression(X1, Y1)
# # 生成拟合直线的绘制点
# print(a0)
# print(a1)
# print(math.atan(a0))
# print(math.degrees(math.atan(a0)))
# x1 = np.arange(0,110,0.01)
# y1 = a0*x1+a1
#
#
# # plt.plot(x1, y1, "blue")
# plt.title("www.jb51.net test")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
#
#
