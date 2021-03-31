import pandas as pd
import numpy as np

# 读取与展示图片
from skimage.io import imread
from PIL import Image
import matplotlib.pyplot as plt
import imageio

# 创建验证集
from sklearn.model_selection import train_test_split

# 评估模型
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Pytorch的相关库
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# your route:
my_route = "/Users/liyuanfeng/Desktop/计算机视觉/messy-vs-clean-room/images/"

train_img = []
for img_num in range(0,96):
    # 定义图像路径
    image_path = my_route+'train/clean/'+str(img_num) + '.png'
    # 读取图片
    img = imread(image_path, as_gray=True)
    # 归一化像素值
    img /= 255.0
    # 转换为浮点数
    img = img.astype('float32')
    # 添加到列表
    train_img.append(img)


plt.imshow(train_img[1],cmap='gray')
plt.show()

for img_num in range(0,96):
    image_path = my_route+'train/messy/'+str(img_num) + '.png'
    img = imread(image_path, as_gray=True)
    img /= 255.0
    img = img.astype('float32')
    train_img.append(img)

# 转换为numpy数组
train_img = np.array(train_img)
train_lable = []
train_lable[:96] = list(1 for i in range(0,96))
# clean
train_lable[96:] = list(0 for i in range(0,96))
train_lable=np.array(train_lable)
# messy
'''
image=Image.open('D:/messy-vs-clean-room/images/test/0.png')
image_arr=np.array(image)
print(image_arr.shape)
'''

'''
i = 0
plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(train_img[i], cmap='gray')
plt.subplot(222), plt.imshow(train_img[i+25], cmap='gray')
plt.subplot(223), plt.imshow(train_img[i+50], cmap='gray')
plt.subplot(224), plt.imshow(train_img[i+75], cmap='gray')
plt.show()
'''

#加载验证集
val_img=[]
for img_num in range(0,10):
    image_path = my_route+'val/clean/'+str(img_num) + '.png'
    img = imread(image_path, as_gray=True)
    img /= 255.0
    img = img.astype('float32')
    val_img.append(img)
for img_num in range(0,10):
    image_path = my_route+'val/messy/'+str(img_num) + '.png'
    img = imread(image_path, as_gray=True)
    img /= 255.0
    img = img.astype('float32')
    val_img.append(img)
val_img = np.array(val_img)
val_lable = []
val_lable[:10] = list(1 for i in range(0,10))
# clean
val_lable[10:] = list(0 for i in range(0,10))
# messy
val_lable=np.array(val_lable)

#测试集
test_img=[]
for img_num in range(0,10):
    image_path = my_route+'test/'+str(img_num) + '.png'
    img = imread(image_path, as_gray=True)
    img /= 255.0
    img = img.astype('float32')
    test_img.append(img)
test_img = np.array(test_img)



train_img = train_img.reshape(192, 1, 299, 299)
train_img = torch.from_numpy(train_img)

# 转换为torch张量
train_label = train_lable.astype(int)
train_label = torch.from_numpy(train_label)

# 转换为torch张量
val_img = val_img.reshape(20, 1, 299, 299)
val_img = torch.from_numpy(val_img)

# 转换为torch张量
val_label = val_lable.astype(int)
val_label = torch.from_numpy(val_label)

test_img = test_img.reshape(10, 1, 299, 299)
test_img = torch.from_numpy(test_img)



def getTrain():
    return train_img, train_lable

def getVal():
    return val_img, val_lable

def getTest():
    return test_img