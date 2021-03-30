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
import torchvision.models as models

import dataset
import my_model
#加载数据集
train_img, train_label = dataset.getTrain()
train_label=torch.from_numpy(train_label)
val_img, val_label = dataset.getVal()
val_label=torch.from_numpy(val_label)
test_img = dataset.getTest()

model = models.resnet18(pretrained=False, num_classes=2)

resnet18 = models.resnet18(pretrained=False)
print(model)


# 定义优化器
optimizer = Adam(model.parameters(), lr=0.07)#lr 学习率
# 定义loss函数
criterion = CrossEntropyLoss()
# 检查GPU是否可用
if torch.cuda.is_available():
    model0 = model.cuda()
    criterion = criterion.cuda()


train_losses = []
val_losses = []
def train(epoch):
    model.train()
    tr_loss = 0
    # 获取训练集

    img_train, label_train = Variable(train_img), Variable(train_label)
    # 获取验证集

    img_val, label_val = Variable(val_img), Variable(val_label)
    # 转换为GPU格式
    if torch.cuda.is_available():
        img_train = img_train.cuda()
        label_train = label_train.cuda()
        img_val = img_val.cuda()
        label_val = label_val.cuda()

    # 清除梯度
    optimizer.zero_grad()

    # 预测训练与验证集
    output_train = model(img_train)
    output_val = model(img_val)

    # 计算训练集与验证集损失
    label_val = torch.tensor(label_val, dtype=torch.long)
    label_train = torch.tensor(label_train, dtype=torch.long)
    loss_train = criterion(output_train, label_train)
    loss_val = criterion(output_val, label_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # 更新权重
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # 输出验证集loss
        print('Epoch : ', epoch+1, '\t', 'loss :', loss_val)

n_epochs = 25
for epoch in range(n_epochs):
    train(epoch)

# 画出loss曲线
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()

#训练集预测
with torch.no_grad():
    output = model(train_img.cuda())
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
# 训练集精度
print(accuracy_score(train_label, predictions))


# 验证集预测
with torch.no_grad():
    output = model(val_img.cuda())
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
# 验证集精度
print(accuracy_score(val_label, predictions))

# 生成测试集预测
with torch.no_grad():
    output = model(test_img.cuda())

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
print(predictions)




