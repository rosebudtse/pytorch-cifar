'''

此文件用来训练，在训练之前确保已进行数据集的切分，分为训练集和测试集

'''




import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
from models import *
from utils import progress_bar

# 额外可设置参数
parser = argparse.ArgumentParser(description='PyTorch JIER Training')
# 初始学习率设置
parser.add_argument('--lr', default=0.001, type=float, help='learning rate') 
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch

#训练总轮数
total_epoch = 200

# 每次送入网络的批次大小
batch_size = 64
# 模型保存的地址
checkpoints_dir = r'./output_new'
accuracies_train=[]
accuracies_test=[]
loss_train=[]
loss_test=[]


# Data
print('==> Preparing data..')
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std_range=(0, 20), probability=0.2):
        self.mean = mean
        self.std_range = std_range
        self.probability = probability

    def __call__(self, image):
        if torch.rand(1) < self.probability:
            std = random.uniform(self.std_range[0], self.std_range[1])
            noise = torch.randn_like(image) * std + self.mean
            image = image + noise
        return image

# 对训练集的图像变换
transform_train = transforms.Compose([
    # 要对图像进行重采样到512*512，原来3000多*3000多的图像太大了，送不进网络
    transforms.Resize((512, 512)),
    # 尾部过大识别率低的罪魁祸首，但是不启用就过拟合，看着办吧
    transforms.RandomCrop(448, padding=32),
    # 图像的方向都一样，就不用随机反转了
#     transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # 随机概率添加随机噪声
    AddGaussianNoise(mean=0.0, std_range=(0, 20), probability=0.2),
    # 归一化，这些值就是我算的极耳数据集的，不用改
    transforms.Normalize((0.2117, 0.2117, 0.2117), (0.2194, 0.2194, 0.2194)),
])

# 对测试集的图像变换
transform_test = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.2117, 0.2117, 0.2117), (0.2194, 0.2194, 0.2194)),
])

# 总的未切分训练集测试集的数据集地址，这个主要是标签从这里读，你也可以改成从训练集或者测试集读，都一样

data_dir = r'./dataset'

#os.removedirs('./jier/.ipynb_checkpoints')

train_data_dir = r'./train_test/train'  # 指定训练集文件夹路径
test_data_dir = r'./train_test/test'  # 指定测试集文件夹路径

# 加载训练集
trainset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=transform_train)

# 加载测试集
testset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=transform_test)


# 创建训练集和测试集的数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# 读取标签，读的是总的数据集的标签
classes = trainset.classes
#classes = ['DaGuang', 'DouDong', 'GuoBao', 'MoHu', 'MoHu_ZuoChao', 'Weibuguoda','WeiYing', 'ZangWu']

print('classes:', classes)


# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()

# 选择网络，你可以改模型文件，添加个B7的config，B3跑得快但是准确率没B7高
net = EfficientNetB3()

#device = 'cpu'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    
# 如果要从检查点继续训练，读取模型
checkpoint_model =  'eff_150.pth'   

criterion = nn.CrossEntropyLoss()
# SGD优化器，可以在后期使用
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
# Adam优化器，不依赖初始学习率，模型训练初期用它收敛的嘎嘎快
optimizer = optim.Adam(net.parameters(), lr = args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,amsgrad=False)
# 更高版本的pytorch可以用下面这个NAdam优化器，更好，但是服务器上pytorch版本太低了用不了，报错的话就老老实实用Adam吧
# optimizer = optim.NAdam(net.parameters(), lr = args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum_decay=0.004, foreach=None, differentiable=False)

# 余弦退火，注意T_max的设置，一般大于等于训练总轮数，要是小于的话，在训练尾声学习率会涨上去
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 继续训练
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoints_dir), 'Error: no checkpoint directory found!'
    checkpoint_path = f'{checkpoints_dir}/{checkpoint_model}'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch'] + 1
        start_epoch = 0
        print(f"Successfully importing {checkpoint_model}!")
    else:
        print("Checkpoint model not found!")

# 模型训练
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    '''
    如果每轮训练都采用80%的训练集开始训练，可以启用这三行代码，会更鲁棒，但是训的慢
    random_sample = random.sample(range(len(trainset)), int(len(trainset)*0.8))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(random_sample)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=train_sampler, num_workers=2)
    '''
    
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):        
        inputs, targets = inputs.to(device), targets.to(device)       
        inputs = inputs.contiguous()       
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
#         print(optimizer)      
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    accuracies_train.append(100.*correct/total)
    loss_train.append(train_loss/(batch_idx+1))

# 模型评估
def test(epoch):
    global best_acc, best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("==> Starting evaluating..")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.5f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    accuracies_test.append(100.*correct/total)
    loss_test.append(test_loss/(batch_idx+1))

    # 保存模型检查点
    loss = test_loss/(batch_idx+1)
    acc = 100.*correct/total
    print('Saving the checkpoint model...')
    # 保存模型的参数，loss, acc, epoch和优化器
    state = {
        'net': net.state_dict(),
        'loss': loss,
        'acc': acc,
        'epoch': epoch,
        'optimizer': optimizer,
    }
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
#     torch.save(state, f'{checkpoints_dir}/best_model.pth')    
    filename = f'{checkpoints_dir}/eff_{epoch}.pth'
    torch.save(state, filename)
    print("Checkpoint model saved!")
    
    #如果准确率最高，就保存为最优模型eff_best_model.pth
    if acc > best_acc:
        print('Saving the best model...')
        torch.save(state, f'{checkpoints_dir}/eff_best_model.pth')
        best_acc = acc
        best_epoch = epoch
        print('Best model saved!\n'
              f'The best model is at epoch {best_epoch} and the accuracy is {best_acc}.')


if __name__ == '__main__':
    for epoch in range(start_epoch, total_epoch):
        train(epoch)
        # 每训练10个epoch评估一次
        if (epoch) % 10 == 0:
            test(epoch)
        scheduler.step()

    # 会生成两个txt，分别存放train和test的loss和acc,相当于日志
    train_file = open('train.txt', 'w')
    test_file = open('test.txt', 'w')
    for i in range(len(loss_train)):
        train_file.write(f'Epoch {i+1}: loss: {loss_train[i]}, acc: {accuracies_train[i]}\n')

    for i in range(len(loss_test)):
        test_file.write(f'Epoch {i*10+1}: loss: {loss_test[i]}, acc: {accuracies_test[i]}\n')

    train_file.close()
    test_file.close()
        
        
