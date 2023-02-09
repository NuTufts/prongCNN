
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

import wandb


parser = argparse.ArgumentParser("train Prong CNN")
parser.add_argument("-t", "--train_file", type=str, default="images/prongCNN_images_file_3particle_train.root", help="train images file")
parser.add_argument("-v", "--val_file", type=str, default="images/prongCNN_images_file_3particle_test.root", help="validation images file")
parser.add_argument("-d", "--device", type=str, default="cuda:0", help="gpu/cpu device")
parser.add_argument("-n", "--num_workers", type=int, default=10, help="number of cpu workers for data loading")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="training batch size")
parser.add_argument("-l", "--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("-p", "--plot_tag", type=str, default="prongCNN", help="tag for output plots")
parser.add_argument("-m", "--model_path", type=str, default="/home/mrosenberg/prongCNN/ResNet18_3part_plAll.pt", help="model name")
args = parser.parse_args()


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

wandb.init(project="prongCNN-ResNet18-test1")


train_transform = transforms.Compose([transforms.Normalize(mean, std),
                                      #transforms.GaussianBlur(kernel_size=(3,3)),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomVerticalFlip(0.5)])

#test_transform = transforms.Compose([transforms.Normalize(mean, std),
#                                     transforms.GaussianBlur(kernel_size=(3,3))])
test_transform = transforms.Normalize(mean, std)

train_dataset = ProngDataset(args.train_file, transformations=train_transform, clip=4.0, threePart=True)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

test_dataset = ProngDataset(args.val_file, transformations=test_transform, clip=4.0, threePart=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


model = ResNet18(1, ResBlock, outputs=3).to(args.device)
print(model)

wandb.watch(model,log="all",log_freq=25)

opt = AdamW(model.parameters(), lr=args.learning_rate)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset.classes),                                                  y=train_dataset.classes)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(args.device)
print("class_weights:", class_weights)

lossFn = nn.NLLLoss(weight=class_weights) #use if softmax is in model
#lossFn = nn.CrossEntropyLoss() #use if softmax not in model


def test(dataloader, model, loss_fn, n_batches=-1):
    
    model.eval()
    
    totalTestLoss = 0
    testCorrect = 0
    testCorrect_e = 0
    testCorrect_pi = 0
    testCorrect_pr = 0
    total_e = 0
    total_pi = 0
    total_pr = 0
    testSteps = len(dataloader.dataset) // dataloader.batch_size
    tstep = 0
    
    with torch.no_grad():
        
        for batch, (X, y) in enumerate(dataloader):
            if n_batches > 0 and tstep >= n_batches:
                break
            if tstep % 100 == 0:
                print("reached validation batch %i of %i"%(tstep, testSteps), flush=True)
            y = y.type(torch.LongTensor)
            X, y = X.to(args.device), y.to(args.device)
            pred = model(X)
            #print(pred)
            #print(y)
            #print(loss_fn(pred, y))
            
            iEl = (y == 0).nonzero(as_tuple=True)
            iPi = (y == 1).nonzero(as_tuple=True)
            iPr = (y == 2).nonzero(as_tuple=True)

            total_e += y[iEl].size(dim=0)
            total_pi += y[iPi].size(dim=0)
            total_pr += y[iPr].size(dim=0)

            totalTestLoss += loss_fn(pred, y)
            testCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            if y[iEl].size(dim=0) > 0:
                testCorrect_e += (pred[iEl].argmax(1) == y[iEl]).type(torch.float).sum().item()
            if y[iPi].size(dim=0) > 0:
                testCorrect_pi += (pred[iPi].argmax(1) == y[iPi]).type(torch.float).sum().item()
            if y[iPr].size(dim=0) > 0:
                testCorrect_pr += (pred[iPr].argmax(1) == y[iPr]).type(torch.float).sum().item()

            tstep += 1
            
    #avgTestLoss = totalTestLoss / testSteps
    #testAcc = testCorrect / len(dataloader.dataset)
    avgTestLoss = totalTestLoss / tstep
    testAcc = testCorrect / (tstep*dataloader.batch_size)
    testAcc_e = testCorrect_e / total_e
    testAcc_pi = testCorrect_pi / total_pi
    testAcc_pr = testCorrect_pr / total_pr
    
    return avgTestLoss.item(), testAcc, testAcc_e, testAcc_pi, testAcc_pr



def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, step, epoch):

    model.train()

    totalTrainLoss = 0
    trainCorrect = 0
    trainSteps = len(train_dataloader.dataset) // train_dataloader.batch_size
    dataloading_time = 0.
    backprop_time = 0.
    
    start = time.time()
    for batch, (X,y) in enumerate(train_dataloader):
        dataloading_time += time.time() - start
        y = y.type(torch.LongTensor)
        X, y = X.to(args.device), y.to(args.device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        start = time.time()
        loss.backward()
        #nn.utils.clip_grad_value_(model.parameters(), clip_value=0.2)
        backprop_time += time.time() - start
        optimizer.step()
        
        totalTrainLoss += loss
        batchCorrect = (pred.argmax(1) == y).type(torch.float).sum().item()
        trainCorrect += batchCorrect
        batchAcc = batchCorrect / train_dataloader.batch_size

        if step % 100 == 0:
            print("reached training batch %i of %i"%(step, trainSteps), flush=True)
            valLoss, valAcc, valAcc_e, valAcc_pi, valAcc_pr = test(test_dataloader, model, loss_fn, 10)
            wandb.log({"train_loss": loss, "train_acc": batchAcc, "val_loss": valLoss, 
                       "val_acc": valAcc, "val_electron_acc": valAcc_e, "val_pion_acc": valAcc_pi,
                       "val_proton_acc": valAcc_pr, "epoch": epoch, "step": step})

        step += 1
        start = time.time()
        
    avgTrainLoss = totalTrainLoss / trainSteps
    trainAcc = trainCorrect / len(train_dataloader.dataset)
    
    print("total time spent loading data:   ", dataloading_time, flush=True)
    print("total time spent doing backprop: ", backprop_time, flush=True)
        
    return step, avgTrainLoss.item(), trainAcc


step = 0

for e in range(1,args.epochs+1):
    step, trL, trA = train(train_dataloader, test_dataloader, model, lossFn, opt, step, e)
    teL, teA, teA_e, teA_pi, teA_pr = test(test_dataloader, model, lossFn)
    print("EPOCH:", e, " train loss:",trL, " train accuracy:", trA, " test loss:", teL, " test accuracy:", teA, " electron test accuracy:", teA_e, " pion test accuracy:", teA_pi, " proton test accuracy:", teA_pr, flush=True)
    torch.save(model.state_dict(), args.model_path.replace(".pt", "_epoch%i.pt"%e))


