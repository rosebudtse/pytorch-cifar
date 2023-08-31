'''

此文件用来制定分类的修正误判

以脏污为例，在gt为脏污的图片被误判成正常的文件中：
如果：
    正常的置信度 < k1
    脏污的脏污置信度 > k2
那么这张被误判为正常的文件就会修正为脏污

这只是一个简单的算法，只要大于小于这两个值就修正，你也可以尝试更复杂的触发条件


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


count = 0
fix_count = 0
z_count = 0
z_count_fix = 0
# 置信度1
k1 = 0.7
# 置信度2
k2 = 0.24
top3_min_daguang = float('-inf')  # 初始化top-3中最高的daguang数值

daguang_value=[]
daguang_percentage = []

daguang_dir = r'train_test/test/zangwu'  # 你想测试的类的文件夹路径
file_count = len(os.listdir(daguang_dir))
for filename in os.listdir(daguang_dir):
    image_path = os.path.join(daguang_dir, filename)
    image = Image.open(image_path)

    # 预处理图片并添加批次维度
    image_tensor = transform(image).unsqueeze(0)

    # 运行模型并计算top-3数值和标签
    output = net(image_tensor)
    top3_values, indices = torch.topk(output, k=3)  # 获取前3个最大的数值和对应的索引
#     print(top3_values)
#     print(indices)
    first_value = top3_values[0][0]
    second_value = top3_values[0][1]   
#     first_labels = [labels[idx] for idx in indices[0][0].item()]
#     second_labels = [labels[idx] for idx in indices[0][1].item()]
    first_label = labels[indices[0][0].item()]
    second_label = labels[indices[0][1].item()]
#     print('first label:',first_label)

    if first_label == 'zangwu':
        count += 1

    '''
    if first_label == 'daguang':#如果识别为正常
        print(f'识别为打光的top3:')
        print(f'{labels[indices[0][0].item()]}: {top3_values[0][0] / torch.sum(top3_values) * 100:.2f}%')
        print(f'{labels[indices[0][1].item()]}: {top3_values[0][1] / torch.sum(top3_values) * 100:.2f}%')        
        print(f'{labels[indices[0][2].item()]}: {top3_values[0][2] / torch.sum(top3_values) * 100:.2f}%')        
    '''
    if first_label == 'zhengchang':#如果识别为正常
        zhengchang_percentage = first_value / torch.sum(top3_values)
        print(f'识别为正常的top3:')
        print(f'{labels[indices[0][0].item()]}: {top3_values[0][0] / torch.sum(top3_values) * 100:.2f}%')
        print(f'{labels[indices[0][1].item()]}: {top3_values[0][1] / torch.sum(top3_values) * 100:.2f}%')        
        print(f'{labels[indices[0][2].item()]}: {top3_values[0][2] / torch.sum(top3_values) * 100:.2f}%')        
        z_count +=1
#         print('first label:',first_label)
        if labels.index('zangwu') in indices[0]:# 如果Top-3中有脏污
#         if second_label == 'zangwu':#判断top-2是否为脏污
            # 找到索引位置
            zangwu_index = torch.nonzero(torch.eq(indices[0], labels.index('zangwu')))[0][0]
            # 获取对应的值
            zangwu_value = top3_values[0][zangwu_index]
            # 计算zangwu_percentage
            zangwu_percentage = zangwu_value / torch.sum(top3_values)
#             zangwu_percentage = top3_values[0][indices[0].index('zangwu')] / torch.sum(top3_values)
#             print(f'second label is zangwu: {zangwu_percentage}')
            if zhengchang_percentage < k1 and zangwu_percentage > k2:#如果top_2是脏污且置信度大于k
                first_label = 'zangwu'#就判别为脏污
                fix_count += 1
                

#         print(f'\n{filename}')
z_count_fix = z_count - fix_count

acc_zangwu = count/file_count * 100
acc_zangwu_fix = (count + fix_count)/file_count * 100
acc_zhengchang = z_count/file_count * 100
acc_zhengchang_fix = (z_count - fix_count)/file_count * 100

print(f'脏污：\n修正前acc: {acc_zangwu:.2f}%，总共{file_count}张，脏污{count}张，正常{z_count}张')
print(f'修正后acc: {acc_zangwu_fix:.2f}%,修正正常{fix_count}张，修正后脏污{count + fix_count}张，正常{z_count_fix}张')

# print(f'正常：\n修正前acc: {acc_zhengchang:.2f}%，总共{file_count}张，脏污{count}张，正常{z_count}张')
# print(f'修正后acc: {acc_zhengchang_fix:.2f}%,修正正常{fix_count}张，修正后脏污{count + fix_count}张，正常{z_count_fix}张')

# print(f'acc: {acc}%')
