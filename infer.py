'''Train JIER with PyTorch.'''
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


parser = argparse.ArgumentParser(description='PyTorch JIER Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch
batch_size = 64
checkpoints_dir = r'./output_newnew'
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


transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
#     transforms.RandomCrop(448, padding=32),
#     transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0.0, std_range=(0, 20), probability=0.2),
    transforms.Normalize((0.2117, 0.2117, 0.2117), (0.2194, 0.2194, 0.2194)),
])

transform_test = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.2117, 0.2117, 0.2117), (0.2194, 0.2194, 0.2194)),
])

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
#net = SimpleDLA()

net = EfficientNetB5()

#device = 'cpu'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint_model =  'eff_120.pth'   

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
optimizer = optim.Adam(net.parameters(), lr = args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,amsgrad=False)
# optimizer = optim.NAdam(net.parameters(), lr = args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum_decay=0.004, foreach=None, differentiable=False)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoints_dir), 'Error: no checkpoint directory found!'
    checkpoint_path = f'{checkpoints_dir}/{checkpoint_model}'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
#         start_epoch = checkpoint['epoch'] + 1
        start_epoch = 0
        print(f"Successfully importing {checkpoint_model}!")
    else:
        print("Checkpoint model not found!")
  



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
#     random_sample = random.sample(range(len(trainset)), int(len(trainset)*0.8))
#     train_sampler = torch.utils.data.sampler.SubsetRandomSampler(random_sample)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=train_sampler, num_workers=2)
    
    
    
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

    # Save checkpoint.
    loss = test_loss/(batch_idx+1)
    acc = 100.*correct/total
    print('Saving the checkpoint model...')
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

    if acc > best_acc:
        print('Saving the best model...')
        torch.save(state, f'{checkpoints_dir}/eff_best_model.pth')
        best_acc = acc
        best_epoch = epoch
        print('Best model saved!\n'
              f'The best model is at epoch {best_epoch} and the accuracy is {best_acc}.')


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+120):
        train(epoch)
        if (epoch) % 10 == 0:
            test(epoch)
        scheduler.step()

    train_file = open('train.txt', 'w')
    test_file = open('test.txt', 'w')

    for i in range(len(loss_train)):
        train_file.write(f'Epoch {i+1}: loss: {loss_train[i]}, acc: {accuracies_train[i]}\n')

    for i in range(len(loss_test)):
        test_file.write(f'Epoch {i*10+1}: loss: {loss_test[i]}, acc: {accuracies_test[i]}\n')

    train_file.close()
    test_file.close()
        
        
