'''

此文件用来读取模型，检查模型的性能和各项参数

'''


import torch

# 指定.pth模型文件的路径
model_path = "./output_new/eff_150.pth"

# 加载模型
model = torch.load(model_path)

loss = model['loss']
acc = model['acc']
epoch = model['epoch']
optimizer = model['optimizer']

print(f"Epoch: {epoch}\nLoss:{loss}\nAcc: {acc}\nOptimizer: {optimizer}")
