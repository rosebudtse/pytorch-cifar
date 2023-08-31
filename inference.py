'''

此文件用来对单张图片进行推理，并输出Top-3分类的置信度

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


parser = argparse.ArgumentParser(description='PyTorch JIER Inference')
parser = argparse.ArgumentParser()

# 图像地址放在这，运行脚本时执行格式为 python inference.py --img_path train_test/test/zhengchang/147.png
parser.add_argument('--img_path', type=str, help='推理图像的路径')
args = parser.parse_args()


checkpoints_dir = r'output_new'
checkpoint_model =  'eff_150.pth'   
best_acc = 0
start_epoch = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = EfficientNetB3()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# 加载模型.
print('==> Loading model..')
assert os.path.isdir(checkpoints_dir), 'Error: no checkpoint directory found!'
checkpoint_path = f'{checkpoints_dir}/{checkpoint_model}'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    # 模型的参数保存在'net'中，只读取net就行，除了net还有epoch，loss，optimizer
    net.load_state_dict(checkpoint['net'])
    print(f"Successfully importing {checkpoint_model}!")
else:
    print("Checkpoint model not found!")

# 模型设定为评估模式
net.eval()
net = net.to(device)

# 图像进行变换
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.2117, 0.2117, 0.2117), (0.2194, 0.2194, 0.2194)),
])

# 读取图像
image_path = args.img_path  
image = Image.open(image_path)  # 使用PIL库加载图像
image = transform(image)  # 进行预处理和转换
image = image.unsqueeze(0)  # 添加一个维度，转换为形状为 [1, 3, 224, 224] 的张量

# 读取数据集中的子文件夹名称作为label
data_dir = r'train_test/test'
all_data = torchvision.datasets.ImageFolder(root=data_dir)
labels = all_data.classes
# label实际上就是以下列表
# labels = ['daguang', 'doudong', 'guoan', 'guobao', 'mohu', 'weibuguoda', 'zangwu', 'zhengchang', 'zuochao']

# label转换为中文
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


# 进行推理
with torch.no_grad():
    outputs = net(image)
    # 输出预测结果，其实就是最大的值（置信度），即最有可能的分类
    _, predicted = torch.max(outputs, 1)
    # 输出Top-3的值，和对应label的索引
    top3_values, top3_indices = torch.topk(outputs, k=3)
    print(f'top3_values: {top3_values}')
    print(f'top3_indices: {top3_indices}')

predicted_label = labels[predicted.item()]

predicted_type = label_map[predicted_label]
print(f'Image:',image_path)
print("预测结果:", predicted_type)
print('置信度：')

# 只取Top-3中的正数，负数就忽略不计了，反正都负数了，分类就不可能是你了
positive_values = top3_values[top3_values >= 0]
total_sum = torch.sum(positive_values).item()

for values, indices in zip(top3_values, top3_indices):
    # 舍弃小于0的数值
    for value, index in zip(values, indices):
        if value >= 0:
            label = list(label_map.keys())[index.item()]
            chinese_label = label_map[label]
            percentage = value.item() / total_sum * 100
            print(f'{chinese_label}: {percentage:.2f}%')
