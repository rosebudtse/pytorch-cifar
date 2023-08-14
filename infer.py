import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from models import *


model = RegNetX_200MF()  # 实例化你的模型类
model.load_state_dict(torch.load(r'E:\users\XieZF02\Downloads\180.pth') , strict=False)  # 加载模型权重
model.eval()  # 设置为评估模式


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

image_path = r'E:\users\XieZF02\Desktop\jier\MoHu\5.png'  # 图像文件路径
image = Image.open(image_path)  # 使用PIL库加载图像
image = transform(image)  # 进行预处理和转换
image = image.unsqueeze(0)  # 添加一个维度，转换为形状为 [1, 3, 224, 224] 的张量

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    
labels = ['DaGuang','DouDong','GuoBao','MoHu','MoHu_ZuoChao']  # 标签列表，与模型输出的类别顺序相对应

label_to_number = {'DaGuang': '打光', 'DouDong': '抖动', 'GuoBao': "过曝", 'MoHu': '模糊', 'MoHu_ZuoChao': '模糊左超'}


predicted_label = labels[predicted.item()]

predicted_type = label_to_number[predicted_label]
print("预测结果:", predicted_type)
