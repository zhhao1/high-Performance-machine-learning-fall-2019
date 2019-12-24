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
from time import perf_counter
from torchsummary import summary

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batchtraining', type=int, default=128, metavar='N', help='batch training size')
parser.add_argument('--gpus', type=int, default=1, metavar='N', help='number of gpus')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
#print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchtraining, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchtraining, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

gpusdeviceids = []
for i in range(args.gpus):
    gpusdeviceids.append(i)

if len(gpusdeviceids) > 1:
    print("Let's use", len(gpusdeviceids), "GPUs!")
else:
    print("Let's use", len(gpusdeviceids), "GPU!")
    torch.cuda.set_device(gpusdeviceid[0])

net = ResNet18()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=gpusdeviceids)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    #print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    #print('\nEpoch: %d' % epoch)
    net.train().to(device)
    train_loss = 0
    correct = 0
    total = 0
    minibatchcount = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        minibatchcount += 1
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs).to(device)
        loss = criterion(outputs, targets).to(device)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()  
    
    if(epoch==4):
        print('\nTrain Epoch: {} \n Average Accuracy: {:.6f} \n Average Epoch Loss: {:.6f}  \n'.format(epoch, 100.0 * correct/total, train_loss/minibatchcount))

def test(epoch):
    global best_acc
    net.eval().to(device)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs).to(device)
            loss = criterion(outputs, targets).to(device)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
#        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 5):
    train(epoch)
    test(epoch)

