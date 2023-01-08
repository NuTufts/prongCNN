
import argparse
import sys
import os

from larlite import larlite
from larlite import larutil
from ublarcvapp import ublarcvapp
from larcv import larcv
from larflow import larflow

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

import wandb

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/models')


parser = argparse.ArgumentParser("train Prong CNN")
parser.add_argument("-t", "--train_file", type=str, default="images/prongCNN_reco_images_file_smallSample_wMasks_train.root", help="train images file")
parser.add_argument("-v", "--val_file", type=str, default="images/prongCNN_reco_images_file_smallSample_wMasks_test.root", help="validation images file")
parser.add_argument("-c", "--l0inChans", type=int, default=2, help="number of input channels for first conv layer")
parser.add_argument("-d", "--device", type=str, default="cuda", help="gpu/cpu device")
parser.add_argument("-n", "--num_workers", type=int, default=12, help="number of cpu workers for data loading")
parser.add_argument("-bt", "--batch_size_train", type=int, default=32, help="training batch size")
parser.add_argument("-bv", "--batch_size_val", type=int, default=32, help="validation batch size")
parser.add_argument("-nbv", "--n_val_batches", type=int, default=10, help="number of batches to process for validation steps")
parser.add_argument("-f", "--log_frequency", type=int, default=100, help="log progress every this number of training steps")
parser.add_argument("-l", "--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("-sE", "--startEpoch", type=int, default=1, help="first epoch number (change if continuing run)")
parser.add_argument("-sTS", "--startTrainStep", type=int, default=0, help="initial training step number (change if continuing run)")
parser.add_argument("-sLS", "--startLogStep", type=int, default=0, help="initial wandb logging step number (change if continuing run)")
parser.add_argument("-sC", "--startCheckpoint", type=str, default="", help="path for model checkpoint to load (change if continuing run)")
parser.add_argument("-p", "--plot_tag", type=str, default="prongCNN", help="tag for output plots")
parser.add_argument("-m", "--model_path", type=str, default="/home/mrosenberg/prongCNN/ResNet34_recoProng_b32_plAll.pt", help="model name")
parser.add_argument("-r", "--runName", type=str, default="DEFAULT", help="wandb run name")
parser.add_argument("--use6class", action="store_true", help="use 6 classes (include other label)")
parser.add_argument("--softLabels", action="store_true", help="use soft labels for loss")
parser.add_argument("--noMask", action="store_true", help="only use prong pixels")
parser.add_argument("--dropLast", action="store_true", help="drop the last training batch in every epoch")
parser.add_argument("--plane2only", action="store_true", help="only use collection plane images")
parser.add_argument("--resnet18", action="store_true", help="use ResNet18 instead of ResNet34")
parser.add_argument("--singleGPU", action="store_true", help="only use one GPU")
args = parser.parse_args()

if args.noMask:
  from models_instanceNorm import ResBlock, ResNet18, ResNet18Pl2, ResNet34, ResNet34Pl2
elif args.l0inChans == 1:
  from models_instanceNorm_reco_1chan import ResBlock, ResNet18, ResNet18Pl2, ResNet34, ResNet34Pl2
elif args.l0inChans == 2:
  from models_instanceNorm_reco_2chan import ResBlock, ResNet18, ResNet18Pl2, ResNet34, ResNet34Pl2
else:
  print("invalid input for --l0inChans (-c) option")
  sys.exit()

if args.use6class and args.softLabels:
  sys.exit("modules not configured for 6 class soft labels")

nClasses = 5
if args.use6class:
    from datasets_reco import ProngDataset, ProngDatasetPl2, mean, std, meanPl2, stdPl2, ProngDatasetNoMask, ProngDatasetPl2NoMask, mean_nm, std_nm, meanPl2_nm, stdPl2_nm
    nClasses = 6
elif args.softLabels:
    from datasets_reco_5ClassSoftLabel import ProngDataset, ProngDatasetPl2, mean, std, meanPl2, stdPl2, ProngDatasetNoMask, ProngDatasetPl2NoMask, mean_nm, std_nm, meanPl2_nm, stdPl2_nm
else:
    from datasets_reco_5ClassHardLabel import ProngDataset, ProngDatasetPl2, mean, std, meanPl2, stdPl2, ProngDatasetNoMask, ProngDatasetPl2NoMask, mean_nm, std_nm, meanPl2_nm, stdPl2_nm

layer0inChans = args.l0inChans
if args.noMask:
  layer0inChans = 1

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

wandb.init(project="prongCNN-5particle-recoProngs")
if args.runName != "DEFAULT":
  wandb.run.name = args.runName
  wandb.run.save()

if args.plane2only:
    img_mean = meanPl2
    img_std = stdPl2
    if args.noMask:
      img_mean = meanPl2_nm
      img_std = stdPl2_nm
    train_transform = transforms.Compose([transforms.Normalize(img_mean, img_std),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.RandomVerticalFlip(0.5)])
    test_transform = transforms.Normalize(img_mean, img_std)

    if args.noMask:
      train_dataset = ProngDatasetPl2NoMask(args.train_file, transformations=train_transform, clip=4.0)
      test_dataset = ProngDatasetPl2NoMask(args.val_file, transformations=test_transform, clip=4.0)
    else:
      train_dataset = ProngDatasetPl2(args.train_file, transformations=train_transform, clip=4.0)
      test_dataset = ProngDatasetPl2(args.val_file, transformations=test_transform, clip=4.0)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, drop_last=args.dropLast, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_val, drop_last=False, shuffle=True, num_workers=args.num_workers)

    if args.resnet18:
        model = ResNet18Pl2(layer0inChans, ResBlock, outputs=nClasses)
    else:
        model = ResNet34Pl2(layer0inChans, ResBlock, outputs=nClasses)
    if not args.singleGPU:
        model = nn.DataParallel(model)

else:
    img_mean = mean
    img_std = std
    if args.noMask:
      img_mean = mean_nm
      img_std = std_nm
    train_transform = transforms.Compose([transforms.Normalize(img_mean, img_std),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.RandomVerticalFlip(0.5)])
    test_transform = transforms.Normalize(img_mean, img_std)

    if args.noMask:
      train_dataset = ProngDatasetNoMask(args.train_file, transformations=train_transform, clip=4.0)
      test_dataset = ProngDatasetNoMask(args.val_file, transformations=test_transform, clip=4.0)
    else:
      train_dataset = ProngDataset(args.train_file, transformations=train_transform, clip=4.0)
      test_dataset = ProngDataset(args.val_file, transformations=test_transform, clip=4.0)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, drop_last=args.dropLast, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_val, drop_last=False, shuffle=True, num_workers=args.num_workers)

    if args.resnet18:
        model = ResNet18(layer0inChans, ResBlock, outputs=nClasses)
    else:
        model = ResNet34(layer0inChans, ResBlock, outputs=nClasses)
    if not args.singleGPU:
        model = nn.DataParallel(model)

model.to(args.device)
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

if args.startCheckpoint != "":
  checkpoint = torch.load(args.startCheckpoint)
  try:
    model.load_state_dict(checkpoint['model_state_dict'])
  except:
    model.module.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(model)

wandb.watch(model,log="all",log_freq=25)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset.classes), y=train_dataset.classes)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(args.device)
print("class_weights:", class_weights)

def softNLLLoss(pred, target):
    return -(class_weights*target*pred).sum() / class_weights.sum()

#class softNLLLoss(nn.Module):
#    def __init__(self, classWeights):
#        self.weights = classWeights
#    def forward(self, pred, target):
#        return -(self.weights*target*pred).sum() / self.weights.sum()
#example with cross entropy loss:
#class SoftCrossEntropyLoss(): #should inhereit from nn.Module?
#   def __init__(self, weights):
#      super().__init__()
#      self.weights = weights
#   def forward(self, y_hat, y):
#      p = nn.functional.log_softmax(y_hat, 1)
#      w_labels = self.weights*y
#      loss = -(w_labels*p).sum() / (w_labels).sum()
#      return loss
#if args.soft5class:
#    lossFn = softNLLLoss(class_weights)
#else:
#    lossFn = nn.NLLLoss(weight=class_weights) #use if softmax is in model

lossFn = nn.NLLLoss(weight=class_weights) #use if softmax is in model
#lossFn = nn.CrossEntropyLoss() #use if softmax not in model


def test(dataloader, model, loss_fn, n_batches=-1):
    
    model.eval()
    
    totalTestLoss = 0
    testCorrect = 0
    testCorrect_e = 0
    testCorrect_ph = 0
    testCorrect_mu = 0
    testCorrect_pi = 0
    testCorrect_pr = 0
    testCorrect_o = 0
    total_e = 0
    total_ph = 0
    total_mu = 0
    total_pi = 0
    total_pr = 0
    total_o = 0
    testSteps = len(dataloader.dataset) // dataloader.batch_size
    tstep = 0
    
    with torch.no_grad():
        
        for batch, (X, y) in enumerate(dataloader):
            if n_batches > 0 and tstep >= n_batches:
                break
            if tstep % args.log_frequency == 0:
                print("reached validation batch %i of %i"%(tstep, testSteps), flush=True)
            if args.softLabels:
                target = y
                y = y.argmax(1)
                target = target.to(args.device)
            else:
                y = y.type(torch.LongTensor)
            X, y = X.to(args.device), y.to(args.device)
            pred = model(X)
            #print(pred)
            #print(y)
            #print(loss_fn(pred, y))
            
            iEl = (y == 0).nonzero(as_tuple=True)
            iPh = (y == 1).nonzero(as_tuple=True)
            iMu = (y == 2).nonzero(as_tuple=True)
            iPi = (y == 3).nonzero(as_tuple=True)
            iPr = (y == 4).nonzero(as_tuple=True)
            iOt = (y == 5).nonzero(as_tuple=True)

            total_e += y[iEl].size(dim=0)
            total_ph += y[iPh].size(dim=0)
            total_mu += y[iMu].size(dim=0)
            total_pi += y[iPi].size(dim=0)
            total_pr += y[iPr].size(dim=0)
            total_o += y[iOt].size(dim=0)

            if args.softLabels:
                totalTestLoss += softNLLLoss(pred, target).detach().item()
            else:
                totalTestLoss += loss_fn(pred, y).detach().item()
            testCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
            if y[iEl].size(dim=0) > 0:
                testCorrect_e += (pred[iEl].argmax(1) == y[iEl]).type(torch.float).sum().item()
            if y[iPh].size(dim=0) > 0:
                testCorrect_ph += (pred[iPh].argmax(1) == y[iPh]).type(torch.float).sum().item()
            if y[iMu].size(dim=0) > 0:
                testCorrect_mu += (pred[iMu].argmax(1) == y[iMu]).type(torch.float).sum().item()
            if y[iPi].size(dim=0) > 0:
                testCorrect_pi += (pred[iPi].argmax(1) == y[iPi]).type(torch.float).sum().item()
            if y[iPr].size(dim=0) > 0:
                testCorrect_pr += (pred[iPr].argmax(1) == y[iPr]).type(torch.float).sum().item()
            if y[iOt].size(dim=0) > 0:
                testCorrect_o += (pred[iOt].argmax(1) == y[iOt]).type(torch.float).sum().item()

            tstep += 1
            
    #avgTestLoss = totalTestLoss / testSteps
    #testAcc = testCorrect / len(dataloader.dataset)
    avgTestLoss = totalTestLoss / tstep
    testAcc = testCorrect / (tstep*dataloader.batch_size)
    testAcc_e = testCorrect_e / total_e
    testAcc_ph = testCorrect_ph / total_ph
    testAcc_mu = testCorrect_mu / total_mu
    testAcc_pi = testCorrect_pi / total_pi
    testAcc_pr = testCorrect_pr / total_pr
    testAcc_o = 0.
    if total_o > 0:
        testAcc_o = testCorrect_o / total_o
    
    return avgTestLoss, testAcc, testAcc_e, testAcc_ph, testAcc_mu, testAcc_pi, testAcc_pr, testAcc_o



def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, step, logStep, epoch):

    model.train()

    totalTrainLoss = 0
    trainCorrect = 0
    trainSteps = len(train_dataloader.dataset) // train_dataloader.batch_size
    dataloading_time = 0.
    backprop_time = 0.
    
    start = time.time()
    for batch, (X,y) in enumerate(train_dataloader):
        dataloading_time += time.time() - start
        if args.softLabels:
            target = y
            y = y.argmax(1)
            target = target.to(args.device)
        else:
            y = y.type(torch.LongTensor)
        X, y = X.to(args.device), y.to(args.device)
        pred = model(X)
        if args.softLabels:
            loss = softNLLLoss(pred, target)
        else:
            loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        start = time.time()
        loss.backward()
        #nn.utils.clip_grad_value_(model.parameters(), clip_value=0.2)
        backprop_time += time.time() - start
        optimizer.step()
        
        lossVal = loss.detach().item()
        totalTrainLoss += lossVal
        batchCorrect = (pred.argmax(1) == y).type(torch.float).sum().item()
        trainCorrect += batchCorrect
        batchAcc = batchCorrect / train_dataloader.batch_size

        if step % args.log_frequency == 0:
            print("reached training batch %i of %i"%(step, trainSteps), flush=True)
            valLoss, valAcc, valAcc_e, valAcc_ph, valAcc_mu, valAcc_pi, valAcc_pr, valAcc_o = test(test_dataloader, model, loss_fn, args.n_val_batches)
            if args.use6class:
                wandb.log({"train_loss": lossVal, "train_acc": batchAcc, "val_loss": valLoss, "val_acc": valAcc,
                           "val_electron_acc": valAcc_e, "val_photon_acc": valAcc_ph, "val_muon_acc": valAcc_mu,
                           "val_pion_acc": valAcc_pi, "val_proton_acc": valAcc_pr, "val_other_acc": valAcc_o, 
                           "epoch": epoch, "step": step}, step=logStep)
            else:
                wandb.log({"train_loss": lossVal, "train_acc": batchAcc, "val_loss": valLoss, "val_acc": valAcc,
                           "val_electron_acc": valAcc_e, "val_photon_acc": valAcc_ph, "val_muon_acc": valAcc_mu,
                           "val_pion_acc": valAcc_pi, "val_proton_acc": valAcc_pr,
                           "epoch": epoch, "step": step}, step=logStep)
            logStep += 1
            model.train()

        step += 1
        start = time.time()
        
    avgTrainLoss = totalTrainLoss / trainSteps
    trainAcc = trainCorrect / len(train_dataloader.dataset)
    
    print("total time spent loading data:   ", dataloading_time, flush=True)
    print("total time spent doing backprop: ", backprop_time, flush=True)
        
    return step, logStep, avgTrainLoss, trainAcc


step = args.startTrainStep
logStep = args.startLogStep

for e in range(args.startEpoch, args.epochs+args.startEpoch):
    step, logStep, trL, trA = train(train_dataloader, test_dataloader, model, lossFn, optimizer, step, logStep, e)
    teL, teA, teA_e, teA_ph, teA_mu, teA_pi, teA_pr, teA_o = test(test_dataloader, model, lossFn)
    print("EPOCH:", e, " train loss:",trL, " train accuracy:", trA, " test loss:", teL, " test accuracy:", teA,
          " electron test accuracy:", teA_e, " photon test accuracy:", teA_ph, " muon test accuracy:", teA_mu,
          " pion test accuracy:", teA_pi, " proton test accuracy:", teA_pr, " other test accuracy:", teA_o, flush=True)
    torch.save({'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()},
               args.model_path.replace(".pt", "_epoch%i.pt"%e))


