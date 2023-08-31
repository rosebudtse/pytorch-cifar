'''

此文件用来求数据集的归一化参数，即各通道的均值和标准差

'''

import os
import random
import numpy as np
from PIL import Image

root_dir = '/opt/ml/input/data/jier2/data/data_all'  # 数据集根目录
class_folders = os.listdir(root_dir)  # 获取所有类别的文件夹

num_samples = 50  # 每个类别的样本数量

image_list = []  # 存储每个图像的像素值

# 遍历每个类别文件夹
for class_folder in class_folders:
    class_path = os.path.join(root_dir, class_folder)
    if not os.path.isdir(class_path):
        continue
    print(f'开始计算{class_folder}')
    # 获取当前类别文件夹中所有图片的路径
    img_files = os.listdir(class_path)
    # 检查，并选择较小的样本数量
    num_samples_current = min(num_samples, len(img_files))
    # 随机选择指定数量的样本
    sampled_img_files = random.sample(img_files, num_samples_current)
    # 遍历选中的样本图片
    for filename in sampled_img_files:
        img_path = os.path.join(class_path, filename)
        img = Image.open(img_path)
        img = img.convert('RGB')  # 将图像转换为RGB格式
        img_array = np.array(img)  # 将图像转换为数组
        image_list.append(img_array)


# 将图像数组存储在一个三维的numpy数组中
image_array = np.stack(image_list)

# 提取每个通道的数据
r_channel = image_array[:, :, :, 0]  # 红色通道
g_channel = image_array[:, :, :, 1]  # 绿色通道
b_channel = image_array[:, :, :, 2]  # 蓝色通道

# 计算每个通道的平均值和标准差
r_mean, r_std = np.mean(r_channel), np.std(r_channel)
g_mean, g_std = np.mean(g_channel), np.std(g_channel)
b_mean, b_std = np.mean(b_channel), np.std(b_channel)

print("Red channel mean:", r_mean)
print("Red channel std:", r_std)
print("Green channel mean:", g_mean)
print("Green channel std:", g_std)
print("Blue channel mean:", b_mean)
print("Blue channel std:", b_std)
