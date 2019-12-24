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

from models_BN import *
from utils import progress_bar


import time
from torchsummary import summary

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--TestBatchSize', type=int, default=100, help='Test batch size')
parser.add_argument('--Data', type=str, default='data', help="location of train and test data")
parser.add_argument('--Gpu', default=True, action='store_true', help='CPU/GPU')
parser.add_argument('--TrainBatchSize', type=int, default=128, help='Train batch size')
parser.add_argument('--Momentum', type=float, default=0.9, help='Training momentum')
parser.add_argument('--TrainNumWorkers', type=int, default=2, help='workers in multiples of 4 for training data')
parser.add_argument('--TestNumWorkers', type=int, default=2, help='workers in multiples of 4 for testing data')
parser.add_argument('--Epochs', type=int, default=5, help='Epochs Training')
parser.add_argument('--Optimizer', type=str, default='sgd', choices=['sgd', 'sgd_nes', 'adagrad', 'adadelta', 'adam'], help='Training optimizer')
parser.add_argument('--WeightDecay', type=float, default=5e-4, help='Weight decay')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and args.Gpu else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.TrainBatchSize, shuffle=True, num_workers=args.TrainNumWorkers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.TestBatchSize, shuffle=False, num_workers=args.TestNumWorkers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
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
#net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

switcher = {
        'sgd': optim.SGD(net.parameters(), lr=args.lr, momentum=args.Momentum, weight_decay=args.WeightDecay),
        'sgd_nes': optim.SGD(net.parameters(), lr=args.lr, momentum=args.Momentum, weight_decay=args.WeightDecay, nesterov=True),
        'adagrad': optim.Adagrad(net.parameters()),
	'adadelta': optim.Adadelta(net.parameters()),
	'adam': optim.Adam(net.parameters())
    }
    
optimizer = switcher.get(args.Optimizer, optim.SGD(net.parameters(), lr=args.lr, momentum=args.Momentum, weight_decay=args.WeightDecay))



#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.Momentum, weight_decay=args.WeightDecay)

# Time measurement of code in c1

DataLoadingAvg = 0.0
MiniBatchComputationTime =0.0
EpochAvgTime = 0.0
DataLoadingTotal= 0.0
MiniBatchCompTimeAvg = 0.0
TrainLossAcc=0.0
PrecisionColl = 0.0
printTotalTime =0.0
TotalEpochDataLoadingTime = 0.0
TotalEpochComputationTime = 0.0
PrecAvgEpoch =0.0
PrecAvgTestEpoch = 0.0
LossAvgEpoch =0.0
LossAvgTestEpoch = 0.0

# Training
def train(epoch):
    global DataLoadingAvg
    global MiniBatchComputationTime
    global EpochAvgTime
    global DataLoadingTotal
    global MiniBatchCompTimeAvg
    global TrainLossAcc
    global PrecisionColl
    global printTotalTime
    global TotalEpochDataLoadingTime
    global TotalEpochComputationTime
    global PrecAvgEpoch
    global LossAvgEpoch
    print('\nEpoch: %d' % epoch)
    startTimeEpoch = time.perf_counter()
    net.train()
    train_loss = 0
    PrecisionCollBatch = 0
    correct = 0
    total = 0
    DataLoadingTotal = 0.0
    MiniBatchCompTimeAvg = 0.0
    dataLoadingStart = time.perf_counter()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        dataLoadingEnd = time.perf_counter()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        minibatchCompTime = time.perf_counter()

        train_loss += loss.item()
        TrainLossAcc += train_loss
        _, predicted = outputs.max(1)
        total += targets.size(0)
        prec = predicted.eq(targets).sum().item()/targets.size(0)
        PrecisionColl  += prec
        PrecisionCollBatch += prec
        correct += predicted.eq(targets).sum().item()
        #minibatchCompTime = time.perf_counter()
        DataLoadingTotal += dataLoadingEnd - dataLoadingStart
        MiniBatchCompTimeAvg += minibatchCompTime - dataLoadingStart
        printStartTime = time.perf_counter()
        print('Train Epoch: {} -  Batch Number: {} - Batch Loss: {} - Batch Data Load Time: {} - Batch Computation Time: {} - Batch precision value: {}  '.format(epoch, batch_idx, loss.item(),(dataLoadingEnd - dataLoadingStart),(minibatchCompTime - dataLoadingStart), prec))
        #print("Batch loss is:")
        #print(loss.item())
        #print("Accumalated batch loss is:")
        #print(train_loss)
        #print("Aggregated batch loss is:")
        #Epoch Loss = train_loss/(batch_idx+1)
        #print("Batch precision value is:")
        #print(prec)
        #print("Each minibatch data loading time")
        #print(dataLoadingEnd - dataLoadingStart)
        #print("Accumalative time for loading minibatch data")
        #print(DataLoadingTotal)
        #print("Aggregate Time for loading minibatch data")
        #print(DataLoadingTotal/(batch_idx+1))
        #print("Each minibatch computation Time")
        #print(minibatchCompTime - dataLoadingStart)
        #print("Accumalative time for  minibatch computation time")
        #print(MiniBatchCompTimeAvg)
        #print("Aggregate time for minibatch Computation Time")
        #print(MiniBatchCompTimeAvg/(batch_idx+1))
        printEndTime = time.perf_counter()
        printTotalTime = printEndTime - printStartTime
        dataLoadingStart = time.perf_counter()
        batchCompTime = time.perf_counter()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    endTimeEpoch = time.perf_counter()
    print('Epoch Number: {} - Total Epoch Time: {} - Total Epoch Loss: {} - Average Epoch Loss: {} - Total Precision Value: {} - Average Precision Value: {}'.format(epoch,(endTimeEpoch - startTimeEpoch- printTotalTime),train_loss,(train_loss/(batch_idx+1)),PrecisionCollBatch, (PrecisionCollBatch/(batch_idx+1))))
    print('Epoch Number: {} - Total Time loading all batches data in this epoch: {} - Average Time loading all batches data in this epoch: {} - Total time for all batches computation in this epoch: {} - Average time for all batches computation in this epoch: {}'.format(epoch, DataLoadingTotal, (DataLoadingTotal/(batch_idx+1)),MiniBatchCompTimeAvg, (MiniBatchCompTimeAvg/(batch_idx+1))))
    #print(epoch)
    #print('Each epoch time')
    #print(endTimeEpoch - startTimeEpoch- printTotalTime)
    EpochAvgTime += (endTimeEpoch - startTimeEpoch- printTotalTime)
    TotalEpochDataLoadingTime += DataLoadingTotal
    TotalEpochComputationTime += MiniBatchCompTimeAvg
    PrecAvgEpoch += (PrecisionCollBatch/(batch_idx+1))
    LossAvgEpoch += (train_loss/(batch_idx+1))
    #print('Accumalative Time for each epoch')
    #print(EpochAvgTime)
    #print('Aggregate Time for each epoch')
    #print(EpochAvgTime/(epoch+1))


def test(epoch):
    global best_acc
    global PrecAvgTestEpoch
    global LossAvgTestEpoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    testPrec_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            prec = correct/targets.size(0)
            print('Test Epoch: {} -  Batch Number: {} - Batch Loss: {} - Batch precision value: {}  '.format(epoch, batch_idx, loss.item(),prec))

           # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
               # % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        PrecAvgTestEpoch += prec/(batch_idx+1)
        LossAvgTestEpoch += test_loss/(batch_idx+1)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, args.Epochs):
    #global EpochAvgTime
    #global TotalEpochDataLoadingTime
    #global TotalEpochComputationTime
    train(epoch)
    print("**************************************************")
    print('Total Epochs time: {} - Average all Epochs Time: {} - All Epoch Data Loading Time: {} - Average all Epochs Data Loading Time: {} - All Epoch Computation Time: {} - Average all Epochs Total Computation Time: {} - All Epoch Loss: {} - Average all Epochs Loss: {} - All Epoch Precision Value: {} - Average all Epochs PrecisionValue: {}'.format(EpochAvgTime,(EpochAvgTime/args.Epochs),TotalEpochDataLoadingTime, (TotalEpochDataLoadingTime/args.Epochs), TotalEpochComputationTime, (TotalEpochComputationTime/args.Epochs),LossAvgEpoch, (LossAvgEpoch/args.Epochs), PrecAvgEpoch, (PrecAvgEpoch/args.Epochs)))
    print("**************************************************")
    test(epoch)
print('Average Test Loss: {} - Average Precision value: {}'.format((PrecAvgTestEpoch/args.Epochs),(LossAvgTestEpoch/args.Epochs)))

summary(net,(3,32,32))

