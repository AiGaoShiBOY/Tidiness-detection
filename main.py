import time

# 读取与展示图片
import matplotlib.pyplot as plt
import numpy as np
# Pytorch的相关库
import torch
import torchvision.models as models
# 评估模型
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.nn import Linear, CrossEntropyLoss, Softmax
from torch.optim import Adam,SGD

import Dataset
import DatasetFor5layer
import my_model

# 创建验证集


# 选取要使用的模型

print("1--------5-layer-CNN")
print("2--------Resnet18")
print("3--------FC-Finetune Pretrained Resnet18")
key = input()

if key == '1':
    train_img, train_label = DatasetFor5layer.getTrain()
    train_label = torch.from_numpy(train_label)
    val_img, val_label = DatasetFor5layer.getVal()
    val_label = torch.from_numpy(val_label)
    test_img = DatasetFor5layer.getTest()
    model = my_model.Net()
    optimizer = Adam(model.parameters(), lr=1e-3)

if key == '2':
    train_img, train_label = Dataset.getTrain()
    train_label = torch.from_numpy(train_label)
    val_img, val_label = Dataset.getVal()
    val_label = torch.from_numpy(val_label)
    test_img = Dataset.getTest()
    model = models.resnet18(pretrained=False, num_classes=2)
    optimizer = SGD(model.parameters(), lr=1e-2)


if key == '3':
    train_img, train_label = Dataset.getTrain()
    train_label = torch.from_numpy(train_label)
    val_img, val_label = Dataset.getVal()
    val_label = torch.from_numpy(val_label)
    test_img = Dataset.getTest()
    model = models.resnet18(pretrained=True)
    model.fc = Linear(in_features=512, out_features=2)
    for para in list(model.parameters())[:-2]:
        para.requires_grad = False
    optimizer = Adam(params=[model.fc.weight, model.fc.bias], lr=1e-3)
    print('the training layer is:')
    for name, param in model.named_parameters():  # 查看可优化的参数有哪些
        if param.requires_grad:
            print(name)



# 定义loss函数
criterion = CrossEntropyLoss()

# 检查GPU是否可用
if torch.cuda.is_available():
    model0 = model.cuda()
    criterion = criterion.cuda()


train_losses = []
val_losses = []

T_accs = []
Val_accs = []

def train(epoch):
    model.train()
    # 获取训练集

    img_train, label_train = Variable(train_img), Variable(train_label)

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
    # label_val = torch.tensor(label_val, dtype=torch.long)
    # label_train = torch.tensor(label_train, dtype=torch.long)
    loss_train = criterion(output_train, label_train)
    loss_val = criterion(output_val, label_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # 更新权重
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    # 输出验证集loss
    train_correct = torch.zeros(1).squeeze()
    train_total = torch.zeros(1).squeeze()
    out = model(train_img)
    pred = torch.argmax(out, 1)
    train_correct += (pred == label_train).sum().float()
    train_total += len(label_train)
    train_acc = (train_correct / train_total).detach().data.numpy()
    T_accs.append(train_acc)
    train_acc_str = 'Train_Accuracy: %f' % ((train_correct / train_total).detach().data.numpy())

    val_correct = torch.zeros(1).squeeze()
    val_total = torch.zeros(1).squeeze()
    out = model(val_img)
    pred = torch.argmax(out, 1)
    val_correct += (pred == label_val).sum().float()
    val_total += len(label_val)
    val_acc = (val_correct / val_total).detach().data.numpy()
    Val_accs.append(val_acc)
    val_acc_str = 'Val_Accuracy: %f' % ((val_correct / val_total).detach().data.numpy())

    print('Epoch : ', epoch + 1, '/', str(n_epochs), ' ', 'loss :', loss_val.data.numpy(), ' ', train_acc_str, ' ', val_acc_str)


n_epochs = 100


print("Start Training...")
t0 = time.time()

for epoch in range(n_epochs):
    train(epoch)

t1 = time.time()

print('Training Complete.')
print('Time cost:', t1-t0, 's')

# 画出loss曲线
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title('loss')
plt.show()

# 画出Acc曲线
plt.plot(T_accs, label='Training Accuracy')
plt.plot(Val_accs, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()


#训练集预测
with torch.no_grad():
    if torch.cuda.is_available():
        output = model(train_img.cuda())
    else:
        output = model(train_img)
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
# 训练集精度
print(accuracy_score(train_label, predictions))


# 验证集预测
with torch.no_grad():
    if torch.cuda.is_available():
        output = model(val_img.cuda())
    else:
        output = model(val_img)
m = Softmax(dim=1)
output = m(output)
prob = list(output.numpy())
predictions = np.argmax(prob, axis=1)
# 验证集精度
print(accuracy_score(val_label, predictions))

# 生成测试集预测
with torch.no_grad():
    if torch.cuda.is_available():
        output = model(test_img.cuda())
    else:
        output = model(test_img)
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
print('The prediction of test is', predictions)




