import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from models import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from PIL import Image




checkpoints_dir = r'./output_new'
checkpoint_model =  'eff_150.pth'   
best_acc = 0
start_epoch = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'\

net = EfficientNetB3()
net.eval()
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
#net.load_state_dict({k.replace('module',''):v for k,v in torch.load(f'{checkpoints_dir}/{checkpoint_model}').items()})

# Load checkpoint.
print('==> Loading model..')
assert os.path.isdir(checkpoints_dir), 'Error: no checkpoint directory found!'
checkpoint_path = f'{checkpoints_dir}/{checkpoint_model}'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict({k.replace('module',''):v for k,v in torch.load(f'{checkpoints_dir}/{checkpoint_model}').items()})

    net.load_state_dict(checkpoint['net'])
#     net = torch.nn.DataParallel(net)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    print(f"Successfully importing {checkpoint_model}!")
else:
    print("Checkpoint model not found!")

    
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.2117, 0.2117, 0.2117), (0.2194, 0.2194, 0.2194)),
])

# image_path = r'data/guobao/11.png'  # 图像文件路径
# image = Image.open(image_path)  # 使用PIL库加载图像
# image = transform(image)  # 进行预处理和转换
# image = image.unsqueeze(0)  # 添加一个维度，转换为形状为 [1, 3, 224, 224] 的张量



data_dir = r'./train_test/test'
# data_dir = r'./zangwu_test'

#os.removedirs('./jier/.ipynb_checkpoints')
all_data = torchvision.datasets.ImageFolder(root=data_dir)    
labels = all_data.classes    
label_map = {
    'daguang': '打光不均',
    'doudong': '抖动',
    'guoan': '过暗',
    'guobao': '过曝',
    'mohu': '模糊',
    'weibuguoda': '尾部过大',
    'zangwu': '棱镜脏污',
    'zhengchang': '正常',
    'zuochao': '左超视野'
}

print(labels)
# label_to_number = {'daguang': '打光不均', 'doudong': '抖动', 'guoan':'过暗','guobao': "过曝", 'mohu': '模糊', 'weibuguoda':'尾部过大','zangwu': '棱镜脏污','zhengchang':'正常','zuochao':'左超视野'}

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = [transform(image) for image in images]  # 使用上面定义的 transform 将图像对象转换为 Tensor
    images = torch.stack(images)  # 将图像 Tensor 堆叠为一个 batch
    labels = torch.tensor(labels)  # 将标签转换为 Tensor
    return images, labels

dataloader = DataLoader(all_data, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)


# 创建混淆矩阵
confusion = torch.zeros(9, 9)
class_losses = torch.zeros(9, device=device)  # 存储每个类别的损失
class_counts = torch.zeros(9, device=device)  # 存储每个类别的样本数量
class_correct_top3 = torch.zeros(9, device=device)  # 统计各个分类的top-3正确分类数量

criterion = nn.CrossEntropyLoss()
running_loss = 0  # 初始化损失

top_k = 3  # 计算top-3准确率
correct_top_k = 0  # 统计top-k正确分类的数量
# 对每个子文件夹进行评估
for i, (images, labels) in enumerate(dataloader):
#     print('Start evaluating..')
    images = images.to(device)  # 将images移动到CUDA设备上
    labels = labels.to(device)  # 将labels移动到CUDA设备上
    # 进行前向传播
    outputs = net(images)
#     print('output is',outputs)
#     _, predicted = torch.max(outputs, 1)
    ts, predicted = torch.max(outputs, 1)
#     print('tensor is', ts)

#     print('predicted is', predicted)
        # 计算损失
    loss = criterion(outputs, labels)
    running_loss += loss.item() * images.size(0)  # 累积损失
 
    for j in range(len(labels)):
        label = labels[j]
        prediction = predicted[j]
        class_idx = labels[j]
        class_losses[class_idx] += loss.item()
        class_counts[class_idx] += 1
        confusion[label][prediction] += 1
        # 计算top-3准确率
        _, top3_predicted = torch.topk(outputs[j], k=3)
        if label in top3_predicted:
            class_correct_top3[label] += 1
        
# 计算准确率
total_acc = torch.sum(confusion.diag()) / torch.sum(confusion)
percentage_acc = total_acc*100

# 计算top-k准确率
top_k_acc = correct_top_k / len(dataloader.dataset) * 100

# 计算平均损失
average_loss = running_loss / len(dataloader.dataset)
class_avg_losses = class_losses / class_counts
class_top3_acc = class_correct_top3 / class_counts * 100
total_top3_acc = torch.sum(class_correct_top3) / torch.sum(class_counts) * 100

# 打印结果
print(confusion)
print('--------------------------')
print('Top-1 Acc:')
for i in range(9):
    class_accuracy = confusion[i][i] * 100 / torch.sum(confusion[i])
    class_loss = running_loss / torch.sum(confusion[i])
    chinese_label = label_map.get(all_data.classes[i])
    print(f"{chinese_label}: Top-1 acc: {class_accuracy.item():.2f}% loss: {class_avg_losses[i].item():.3f}")
print(f"\n总体准确率: {percentage_acc.item():.2f}%")
print(f"平均损失: {average_loss:.3f}")


print('\n--------------------------')
print('Top-3 Acc:')
for i in range(9):
    class_top3_accuracy = class_top3_acc[i]
    chinese_label = label_map.get(all_data.classes[i])
    print(f"{chinese_label}: Top-3 acc: {class_top3_accuracy.item():.2f}%")
print(f"\n总体准确率: {total_top3_acc.item():.2f}%")


print('\n--------------------------')
print('异常识别成功率：')
success = 0
total = 0
for i in range(9):
    chinese_label = label_map.get(all_data.classes[i])
    if i != 7:
        success_rate = (torch.sum(confusion[i]) - confusion[i][7]) * 100 / torch.sum(confusion[i])
        success += torch.sum(confusion[i]) - confusion[i][7]
        total += torch.sum(confusion[i])
        print(f"{chinese_label}: 识别准确率: {success_rate:.2f}%")
    if i == 7:
        success_rate = confusion[i][i]*100 / torch.sum(confusion[i])
        success += confusion[i][i]
        total += torch.sum(confusion[i])        
        print(f"{chinese_label}: 识别准确率: {success_rate:.2f}%")
        
print(f'\n总体准确率：{success / total * 100:.2f}%')
