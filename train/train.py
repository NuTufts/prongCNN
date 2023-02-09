
import argparse
import sys
import os

import numpy as np

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import time
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/models')

from datasets import ProngDataset, ProngDatasetPl2, mean, std, meanPl2, stdPl2
from models import ResBlock, ResNet18


parser = argparse.ArgumentParser("train Prong CNN")
parser.add_argument("-t", "--train_file", type=str, default="prongCNN_images_file_3particle_train.root", help="train images file")
parser.add_argument("-v", "--val_file", type=str, default="prongCNN_images_file_3particle_test.root", help="validation images file")
parser.add_argument("-d", "--device", type=str, default="cuda:0", help="gpu/cpu device")
parser.add_argument("-n", "--num_workers", type=int, default=10, help="number of cpu workers for data loading")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="training batch size")
parser.add_argument("-l", "--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("-e", "--epochs", type=int, default=20, help="number of training epochs")
parser.add_argument("-p", "--plot_tag", type=str, default="prongCNN", help="tag for output plots")
args = parser.parse_args()


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


train_transform = transforms.Compose([transforms.Normalize(meanPl2, stdPl2),
                                      #transforms.GaussianBlur(kernel_size=(3,3)),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomVerticalFlip(0.5)])

#test_transform = transforms.Compose([transforms.Normalize(mean, std),
#                                     transforms.GaussianBlur(kernel_size=(3,3))])
test_transform = transforms.Normalize(meanPl2, stdPl2)

train_dataset = ProngDatasetPl2(args.train_file, train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

test_dataset = ProngDatasetPl2(args.val_file, test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


model = ResNet18(1, ResBlock, outputs=3).to(args.device)
print(model)

opt = AdamW(model.parameters(), lr=args.learning_rate)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset.classes),                                                  y=train_dataset.classes)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(args.device)
print("class_weights:", class_weights)

lossFn = nn.NLLLoss(weight=class_weights) #use if softmax is in model
#lossFn = nn.CrossEntropyLoss() #use if softmax not in model


def train(dataloader, model, loss_fn, optimizer):

    model.train()

    totalTrainLoss = 0
    trainCorrect = 0
    trainSteps = len(dataloader.dataset) // dataloader.batch_size
    dataloading_time = 0.
    backprop_time = 0.
    step = 0
    
    start = time.time()
    for batch, (X,y) in enumerate(dataloader):
        if step % 100 == 0:
            print("reached training batch %i of %i"%(step, trainSteps), flush=True)
        dataloading_time += time.time() - start
        y = y.type(torch.LongTensor)
        X, y = X.to(args.device), y.to(args.device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        start = time.time()
        loss.backward()
        backprop_time += time.time() - start
        optimizer.step()
        
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        start = time.time()
        step += 1
        
    avgTrainLoss = totalTrainLoss / trainSteps
    trainAcc = trainCorrect / len(dataloader.dataset)
    
    print("total time spent loading data:   ", dataloading_time, flush=True)
    print("total time spent doing backprop: ", backprop_time, flush=True)
        
    return avgTrainLoss.item(), trainAcc


def test(dataloader, model, loss_fn):
    
    model.eval()
    
    totalTestLoss = 0
    testCorrect = 0
    testSteps = len(dataloader.dataset) // dataloader.batch_size
    step = 0
    
    with torch.no_grad():
        
        for batch, (X, y) in enumerate(dataloader):
            if step % 100 == 0:
                print("reached validation batch %i of %i"%(step, testSteps), flush=True)
            y = y.type(torch.LongTensor)
            X, y = X.to(args.device), y.to(args.device)
            pred = model(X)
            #print(pred)
            #print(y)
            #print(loss_fn(pred, y))
            
            totalTestLoss += loss_fn(pred, y)
            testCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            step += 1
            
    avgTestLoss = totalTestLoss / testSteps
    testAcc = testCorrect / len(dataloader.dataset)
    
    return avgTestLoss.item(), testAcc


epoch_list = []
train_loss = []
train_acc = []
test_loss = []
test_acc = []

trL, trA = test(train_dataloader, model, lossFn)
teL, teA = test(test_dataloader, model, lossFn)
print("EPOCH:", 0, " train loss:",trL, " train accuracy:", trA, " test loss:", teL, " test accuracy:", teA, flush=True)
epoch_list.append(0)
train_loss.append(trL)
train_acc.append(trA)
test_loss.append(teL)
test_acc.append(teA)


for e in range(1,args.epochs+1):
    trL, trA = train(train_dataloader, model, lossFn, opt)
    teL, teA = test(test_dataloader, model, lossFn)
    print("EPOCH:", e, " train loss:",trL, " train accuracy:", trA, " test loss:", teL, " test accuracy:", teA, flush=True)
    epoch_list.append(e)
    train_loss.append(trL)
    train_acc.append(trA)
    test_loss.append(teL)
    test_acc.append(teA)


plt.figure(2)
plt.plot(epoch_list, train_loss, 'b-', label='training sample')
plt.plot(epoch_list, test_loss, 'r-', label='test sample')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.yscale("log")
plt.legend()
plt.savefig(args.plot_tag+"_loss_curves.png", format=png)

plt.figure(3)
plt.plot(epoch_list, train_acc, 'b-', label='training sample')
plt.plot(epoch_list, test_acc, 'r-', label='test sample')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(args.plot_tag+"_accuracy_curves.png", format=png)


