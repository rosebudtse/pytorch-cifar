'''

试图寻找一个置信度阈值解决误判问题


可以计算指定分类判定为某一分类的top-1数值的最小值
当然你也可以修改成求其他值，不一定是最小值

'''



from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from PIL import Image


dir = r'train_test/test'
checkpoints_dir = r'output_new'
checkpoint_model =  'eff_150.pth'   

net = EfficientNetB3()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

net.eval()
net = net.to(device)


# Load checkpoint.
print('==> Loading model..')
assert os.path.isdir(checkpoints_dir), 'Error: no checkpoint directory found!'
checkpoint_path = f'{checkpoints_dir}/{checkpoint_model}'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)


    net.load_state_dict(checkpoint['net'])
    print(f"Successfully importing {checkpoint_model}!")
else:
    print("Checkpoint model not found!")

device = 'cuda' if torch.cuda.is_available() else 'cpu'\

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

net.eval()
net = net.to(device)

labels = ['daguang', 'doudong', 'guoan', 'guobao', 'mohu', 'weibuguoda', 'zangwu', 'zhengchang', 'zuochao']

#我想对daguang文件夹下的所有图片进行以下操作：计算每张图片的top-3数值和对应的label，对daguang文件夹中所有top-3中最高的数值属于daguang的图片进行计算：计算他们daguang数值的最小值，然后输出这个最小值
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.2117, 0.2117, 0.2117), (0.2194, 0.2194, 0.2194)),
])


def compute_top3_values(model_output):
    top3_values, indices = torch.topk(model_output, k=3)  # 获取前3个最大的数值和对应的索引
    top3_values = top3_values[0]
    top3_labels = [labels[idx] for idx in indices[0]]
    return top3_values, top3_labels


count = 0
top3_min_daguang = float('-inf')  # 初始化top-3中最高的daguang数值

daguang_value=[]
daguang_percentage = []

daguang_dir = r'train_test/test/zangwu'  # daguang文件夹路径
file_count = len(os.listdir(daguang_dir))
for filename in os.listdir(daguang_dir):
    image_path = os.path.join(daguang_dir, filename)
    image = Image.open(image_path)

    # 预处理图片并添加批次维度
    image_tensor = transform(image).unsqueeze(0)

    # 运行模型并计算top-3数值和标签
    output = net(image_tensor)
    top3_values, top3_labels = compute_top3_values(output)

    # 更新最大值在top-3数值中的占比
    if top3_labels[0] == 'zangwu':
        count += 1
        max_value = torch.max(top3_values)
        max_value_percentage = max_value / torch.sum(top3_values)
        print(max_value_percentage)
        daguang_percentage.append(max_value_percentage)

top3_min_daguang = min(daguang_percentage)
        
#         if max_value > top3_max_daguang:
#             top3_max_daguang = max_value
acc = count/file_count * 100
# 输出top-3中最高的daguang数值的最小值
print(f'Minimum daguang value in top-3: {top3_min_daguang}')
print(f'acc:{acc}%')
