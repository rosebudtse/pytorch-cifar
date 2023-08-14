'''Train CIFAR10 with PyTorch.'''
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


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_dir = './jier'
all_data = torchvision.datasets.ImageFolder(root=data_dir, transform=transform_train)
train_size = int(0.8 * len(all_data))
test_size = len(all_data) - train_size
trainset, testset = torch.utils.data.random_split(all_data, [train_size, test_size])
testset = [(i[0], i[1]) for i in testset]


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

'''
Just like the code from line 28 to line 50, we need to use the dataset from local folder.
and every folder in the upper folder "dataset" is a class, and the pictures in the folder is the samples.
实现和上述代码一样的功能, 只是数据集从main.py同文件夹下的data文件夹中读取, 每个文件夹是一个类，文件夹中的图片是样本。，且变量名字与上述代码一致。
'''





#classes = all_data.classes

classes = ['DaGuang', 'DouDong', 'GuoBao', 'MoHu', 'MoHu_ZuoChao', 'Weibuguoda','WeiYing', 'ZangWu']


#.ipynb_checkpoints文件夹是Jupyter Notebook在保存Notebook时自动生成的文件夹，它记录了Notebook的检查点信息。这个文件夹通常不包含任何图像数据，并且在加载图像文件夹数据集时会被错误地视为一个类别。
if '.ipynb_checkpoints' in classes:
    classes.remove('.ipynb_checkpoints')
    


print(classes)


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

net = RegNetX_200MF()

#device = 'cpu'
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint_path = './checkpoint/150.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        print("Successfully importing model!")
    else:
        print("Checkpoint model not found!")
  
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        inputs = inputs.contiguous() 
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc, best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    losses = []
    accuracies = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            losses.append(test_loss / (batch_idx+1))
            accuracies.append(100. * correct / total)

    # Save checkpoint.
    acc = 100.*correct/total
    print('Saving the checkpoint model...')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/best_model.pth')    
    filename = f'./checkpoint/{epoch}.pth'
    torch.save(state, filename)
    print("Checkpoint model saved!")

    if acc > best_acc:
        print('Saving the best model...')
        torch.save(state, './checkpoint/best_model.pth')
        best_acc = acc
        best_epoch = epoch
        print('Best model saved!\n'
              f'The best model is at epoch {best_epoch} and the accuracy is {best_acc}.')
    return losses, accuracies

if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch+999):
        #print('Start training...')
        train(epoch)
        if (epoch) % 10 == 0:
            print('Start evaluing...')
            test(epoch)
        scheduler.step()
