import os
import random
import shutil

# 设置随机种子，以确保每次运行结果一致
random.seed(42)

# 指定数据集文件夹路径
dataset_folder = r"e:\users\XieZF02\Desktop\dataset"

train_set_folder = r'e:\users\XieZF02\Desktop\train_test'

# 指定训练集和测试集比例
train_ratio = 0.8
test_ratio = 0.2

# 创建train和test文件夹
train_folder = os.path.join(train_set_folder, "train")
os.makedirs(train_folder, exist_ok=True)
test_folder = os.path.join(train_set_folder, "test")
os.makedirs(test_folder, exist_ok=True)

# 遍历A文件夹下的子文件夹（类别）
for category in os.listdir(dataset_folder):
    category_folder = os.path.join(dataset_folder, category)
    if not os.path.isdir(category_folder):
        continue

    # 创建类别对应的train和test子文件夹
    train_category_folder = os.path.join(train_folder, category)
    os.makedirs(train_category_folder, exist_ok=True)
    test_category_folder = os.path.join(test_folder, category)
    os.makedirs(test_category_folder, exist_ok=True)

    # 获取当前类别下的所有图片文件
    image_files = [
        file for file in os.listdir(category_folder) if file.endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    # 根据训练集和测试集比例，计算分割点索引
    split_index = int(len(image_files) * train_ratio)

    # 随机打乱图片文件列表
    random.shuffle(image_files)

    # 将图片按照比例分配到训练集和测试集
    train_images = image_files[:split_index]
    test_images = image_files[split_index:]

    # 将训练集的图片复制到train对应子文件夹中
    for train_image in train_images:
        src_path = os.path.join(category_folder, train_image)
        dst_path = os.path.join(train_category_folder, train_image)
        shutil.copy(src_path, dst_path)

    # 将测试集的图片复制到test对应子文件夹中
    for test_image in test_images:
        src_path = os.path.join(category_folder, test_image)
        dst_path = os.path.join(test_category_folder, test_image)
        shutil.copy(src_path, dst_path)
