
import argparse
import sys
import os
import gc

import numpy as np

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import time
import random

from math import sqrt, log10

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CyclicLR, CosineAnnealingWarmRestarts, LambdaLR
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
parser.add_argument("-l", "--learning_rate", type=float, default=1e-3, help="learning rate (constant or initial for Step, OneCycle, CosinAnnealingWarmRestarts, and custom cyclic with log_triangular and log_triangular2 LR schedulers)")
parser.add_argument("--schedStepLR", action="store_true", help="use a step learning rate scheduler")
parser.add_argument("--schedCyclicLR", action="store_true", help="use a cyclic learning rate scheduler")
parser.add_argument("--schedCosAnnealWRLR", action="store_true", help="use cosine annealing warm restarts learning rate scheduler")
parser.add_argument("--schedOneCycleLR", action="store_true", help="use one cycle with cosine annealing learning rate scheduler")
parser.add_argument("-lrS", "--schedLRStepSize", type=int, default=10, help="set cycle/period IN EPOCHS for change in learning rate when using a learning rate scheduler option")
parser.add_argument("-lrB", "--schedLRBase", type=float, default=1e-8, help="base (minimum) learning rate for oscillatory LR schedulers")
parser.add_argument("-lrM", "--schedLRMax", type=float, default=1e-2, help="maximum learning rate for oscillatory and one cycle LR schedulers")
parser.add_argument("-slrG", "--stepLRGamma", type=float, default=0.1, help="decrease learning rate by this factor after schedLRStepSize epochs with step LR scheduler")
parser.add_argument("-clrM", "--cyclicLRMode", type=str, default="triangular", help="mode for cyclic learning rate scheduler (triangular, triangular2, and exp_range for built-in pytorch schedulers; log_triangular, log_triangular2 for custom")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("-sE", "--startEpoch", type=int, default=1, help="first epoch number (change if continuing run)")
parser.add_argument("-sTS", "--startTrainStep", type=int, default=0, help="initial training step number (change if continuing run)")
parser.add_argument("-sLS", "--startLogStep", type=int, default=0, help="initial wandb logging step number (change if continuing run)")
parser.add_argument("-sC", "--startCheckpoint", type=str, default="", help="path for model checkpoint to load (change if continuing run)")
parser.add_argument("-m", "--model_path", type=str, default="/home/mrosenberg/prongCNN/ResNet34_recoProng_b32_plAll.pt", help="model name")
parser.add_argument("-p", "--projectName", type=str, default="prongCNN-5particle-recoProngs-multiTask", help="wandb project name")
parser.add_argument("-r", "--runName", type=str, default="DEFAULT", help="wandb run name")
parser.add_argument("-wLP", "--partLossWeight", type=float, default=0.5, help="weight for particle classification in multi task loss (must specify --multiTask and --hardWeights)")
parser.add_argument("-dop", "--dropoutProb", type=float, default=0.2, help="dropout probability for class./reg. MLPs (must specify --multiTask and --deepMLPwDO")
parser.add_argument("--multiTask", action="store_true", help="do particle classification and completeness regression")
parser.add_argument("--tripleTask", action="store_true", help="do particle classification and completeness and purity regression")
parser.add_argument("--deepMLP", action="store_true", help="use 3 layer MLPs for output tasks")
parser.add_argument("--deepMLPwBN", action="store_true", help="use 3 layer MLPs with batch norm for output tasks")
parser.add_argument("--deepMLPwIN", action="store_true", help="use 3 layer MLPs with instance norm for output tasks")
parser.add_argument("--deepMLPwDO", action="store_true", help="use 3 layer MLPs with dropout for output tasks")
parser.add_argument("--classifyComp", action="store_true", help="do classification instead of regression for completeness")
parser.add_argument("--hardWeights", action="store_true", help="use hard coded task weights for multi task loss")
parser.add_argument("--use6class", action="store_true", help="use 6 classes (include other label)")
parser.add_argument("--softLabels", action="store_true", help="use soft labels for loss")
parser.add_argument("--noMask", action="store_true", help="only use prong pixels")
parser.add_argument("--keepLast", action="store_true", help="don't drop the last training batch in every epoch. this could mess up the learning rate schedulers")
parser.add_argument("--plane2only", action="store_true", help="only use collection plane images")
parser.add_argument("--resnet18", action="store_true", help="use ResNet18 instead of ResNet34")
parser.add_argument("--singleGPU", action="store_true", help="only use one GPU")
parser.add_argument("--noLogs", action="store_true", help="don't upload to wandb")
args = parser.parse_args()

dropLast = not args.keepLast

step = args.startTrainStep
logStep = args.startLogStep

if args.multiTask and (args.l0inChans != 2 or args.use6class or args.softLabels or args.noMask or args.plane2only or args.resnet18):
  sys.exit("multiTask training only configured for 5 class hard labels with mask (3 plane, 2 in channel config.) with ResNet34")

if args.tripleTask and (args.l0inChans != 2 or args.use6class or args.softLabels or args.noMask or args.plane2only or args.resnet18 or args.hardWeights or args.multiTask):
  sys.exit("tripleTask training only configured for 5 class hard labels with mask (3 plane, 2 in channel config.) with ResNet34 and learnable loss weights")

if ((args.deepMLP or args.deepMLPwBN or args.deepMLPwIN or args.deepMLPwDO) and not args.multiTask) or ((args.deepMLP or args.deepMLPwBN or args.deepMLPwIN or args.deepMLPwDO) and (args.classifyComp or args.hardWeights)):
  sys.exit("deepMLP options are only implemented for multi task config. with completeness regression and learnable loss weights")

if args.tripleTask:
  from models_instanceNorm_reco_2chan_tripleTask import ResBlock, ResNet34
elif args.multiTask:
  from models_instanceNorm_reco_2chan_multiTask import ResBlock, ResNet34, ResNet34ClCmp, ResNet34DeepMLP, ResNet34DeepMLPwBN, ResNet34DeepMLPwIN, ResNet34DeepMLPwDO
elif args.noMask:
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
if args.tripleTask:
  from datasets_reco_5ClassHardLabel_tripleTask import ProngDataset, mean, std
elif args.multiTask:
  from datasets_reco_5ClassHardLabel_multiTask import ProngDataset, ProngDatasetClCmp, mean, std
elif args.use6class:
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


if not args.noLogs:
  wandb.init(project=args.projectName)
  if args.runName != "DEFAULT":
    wandb.run.name = args.runName
    #wandb.run.save()

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
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, drop_last=dropLast, shuffle=True, num_workers=args.num_workers)
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
      if args.multiTask and args.classifyComp:
        train_dataset = ProngDatasetClCmp(args.train_file, transformations=train_transform, clip=4.0)
        test_dataset = ProngDatasetClCmp(args.val_file, transformations=test_transform, clip=4.0)
      else:
        train_dataset = ProngDataset(args.train_file, transformations=train_transform, clip=4.0)
        test_dataset = ProngDataset(args.val_file, transformations=test_transform, clip=4.0)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, drop_last=dropLast, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_val, drop_last=False, shuffle=True, num_workers=args.num_workers)

    if args.resnet18:
        model = ResNet18(layer0inChans, ResBlock, outputs=nClasses)
    else:
        if args.multiTask:
            if args.classifyComp:
                model = ResNet34ClCmp(layer0inChans, ResBlock, outputs=nClasses)
            elif args.deepMLP:
                model = ResNet34DeepMLP(layer0inChans, ResBlock, outputs=nClasses)
            elif args.deepMLPwBN:
                model = ResNet34DeepMLPwBN(layer0inChans, ResBlock, outputs=nClasses)
            elif args.deepMLPwIN:
                model = ResNet34DeepMLPwIN(layer0inChans, ResBlock, outputs=nClasses)
            elif args.deepMLPwDO:
                model = ResNet34DeepMLPwDO(layer0inChans, ResBlock, outputs=nClasses, dropoutProb=args.dropoutProb)
            else:
                model = ResNet34(layer0inChans, ResBlock, outputs=nClasses)
        else:
            model = ResNet34(layer0inChans, ResBlock, outputs=nClasses)
    if not args.singleGPU:
        model = nn.DataParallel(model)

model.to(args.device)
print(model)

if not args.noLogs:
  wandb.watch(model,log="all",log_freq=25)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset.classes), y=train_dataset.classes)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(args.device)
print("class_weights:", class_weights)

if args.multiTask and args.classifyComp:
  comp_class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_dataset.compClasses), y=train_dataset.compClasses)
  comp_class_weights = torch.tensor(comp_class_weights, dtype=torch.float).to(args.device)
  print("comp_class_weights:", comp_class_weights)
  lossFnComp = nn.NLLLoss(weight=comp_class_weights)

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

lossFn = nn.NLLLoss(weight=class_weights) #use if softmax is in model
#lossFn = nn.CrossEntropyLoss() #use if softmax not in model

lossMSEcomp = nn.MSELoss()
lossMSEpur = nn.MSELoss()

class MultiTaskLossHardWeight(nn.Module):
  def __init__(self):
    super(MultiTaskLossHardWeight, self).__init__()
    self.w_class = args.partLossWeight
    self.w_reg = (1.0 - args.partLossWeight)
  def forward(self, outputs, targets):
    loss_class = lossFn(outputs[0], targets[0])
    loss_reg = lossMSEcomp(outputs[1], targets[1])
    loss_total = self.w_class*loss_class + self.w_reg*loss_reg
    return [loss_class, loss_reg], loss_total, [self.w_class, self.w_reg]

class MultiTaskLossClCmp(nn.Module):
  def __init__(self, wC=0.5, wR=0.5):
    super(MultiTaskLossClCmp, self).__init__()
    self.etaC = nn.Parameter(torch.Tensor([wC]))
    self.etaR = nn.Parameter(torch.Tensor([wR]))
  def forward(self, outputs, targets):
    loss_class = lossFn(outputs[0], targets[0])
    loss_comp = lossFnComp(outputs[1], targets[1])
    loss_total = 2.0*torch.exp(-self.etaC)*loss_class + 2.0*torch.exp(-self.etaR)*loss_comp + self.etaC + self.etaR
    return [loss_class, loss_comp], loss_total, [self.etaC, self.etaR]

class MultiTaskLoss(nn.Module):
  def __init__(self, wC=0.5, wR=0.5):
    super(MultiTaskLoss, self).__init__()
    self.etaC = nn.Parameter(torch.Tensor([wC]))
    self.etaR = nn.Parameter(torch.Tensor([wR]))
  def forward(self, outputs, targets):
    loss_class = lossFn(outputs[0], targets[0])
    loss_reg = lossMSEcomp(outputs[1], targets[1])
    loss_total = 2.0*torch.exp(-self.etaC)*loss_class + torch.exp(-self.etaR)*loss_reg + self.etaC + self.etaR
    return [loss_class, loss_reg], loss_total, [self.etaC, self.etaR]

class TripleTaskLoss(nn.Module):
  def __init__(self, wC=0.5, wRc=0.5, wRp=0.5):
    super(TripleTaskLoss, self).__init__()
    self.etaC = nn.Parameter(torch.Tensor([wC]))
    self.etaRc = nn.Parameter(torch.Tensor([wRc]))
    self.etaRp = nn.Parameter(torch.Tensor([wRp]))
  def forward(self, outputs, targets):
    loss_class = lossFn(outputs[0], targets[0])
    loss_comp = lossMSEcomp(outputs[1], targets[1])
    loss_pur = lossMSEpur(outputs[2], targets[2])
    loss_total = 2.0*torch.exp(-self.etaC)*loss_class + torch.exp(-self.etaRc)*loss_comp + torch.exp(-self.etaRp)*loss_pur + self.etaC + self.etaRc + self.etaRp
    return [loss_class, loss_comp, loss_pur], loss_total, [self.etaC, self.etaRc, self.etaRp]


initialLossWeights = [0.5, 0.5, 0.5]
if args.startCheckpoint != "":
  checkpoint = torch.load(args.startCheckpoint)
  if args.multiTask:
    initialLossWeights[0] = checkpoint['weight_state_dict']['etaC']
    initialLossWeights[1] = checkpoint['weight_state_dict']['etaR']
  if args.tripleTask:
    initialLossWeights[0] = checkpoint['weight_state_dict']['etaC']
    initialLossWeights[1] = checkpoint['weight_state_dict']['etaRc']
    initialLossWeights[2] = checkpoint['weight_state_dict']['etaRp']

if args.tripleTask:
  lossMulti = TripleTaskLoss(initialLossWeights[0], initialLossWeights[1], initialLossWeights[2]).to(args.device)

if args.multiTask:
  if args.hardWeights:
    lossMulti = MultiTaskLossHardWeight().to(args.device)
  elif args.classifyComp:
    lossMulti = MultiTaskLossClCmp(initialLossWeights[0], initialLossWeights[1]).to(args.device)
  else:
    lossMulti = MultiTaskLoss(initialLossWeights[0], initialLossWeights[1]).to(args.device)

if (args.multiTask and not args.hardWeights) or args.tripleTask:
  optimizer = AdamW(list(lossMulti.parameters())+list(model.parameters()), lr=args.learning_rate)
else:
  optimizer = AdamW(model.parameters(), lr=args.learning_rate)


if args.startCheckpoint != "":
  try:
    model.load_state_dict(checkpoint['model_state_dict'])
  except:
    model.module.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

useNonStepScheduler = args.schedCyclicLR or args.schedCosAnnealWRLR or args.schedOneCycleLR
itersPerEpoch = len(train_dataloader.dataset) // train_dataloader.batch_size
lrStepSize = args.schedLRStepSize * itersPerEpoch
if args.schedStepLR:
  scheduler = StepLR(optimizer, step_size=args.schedLRStepSize, gamma=args.stepLRGamma)
elif args.schedCyclicLR:
  if args.cyclicLRMode in ["log_triangular","log_triangular2"]:
    ascending = False
    nCycles = args.epochs // (args.schedLRStepSize*2)
    schedLRMax = args.schedLRMax
    schedLRBase = args.schedLRBase
    lambda1 = lambda x: 10**(((log10(schedLRMax/schedLRBase))/lrStepSize)*(x % lrStepSize)) if ascending else               10**((-(log10(schedLRMax/schedLRBase))/lrStepSize)*(x % lrStepSize) + log10(schedLRMax/schedLRBase))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
  else:
    scheduler = CyclicLR(optimizer, mode=args.cyclicLRMode, step_size_up=lrStepSize, base_lr=args.schedLRBase, max_lr=args.schedLRMax, cycle_momentum=False)
elif args.schedCosAnnealWRLR:
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=lrStepSize, eta_min=args.schedLRBase)
elif args.schedOneCycleLR:
  scheduler = OneCycleLR(optimizer, max_lr=args.schedLRMax, steps_per_epoch=itersPerEpoch, epochs=args.epochs, anneal_strategy='cos')



def test(dataloader, n_batches=-1):
    
    model.eval()
    if args.multiTask or args.tripleTask:
        lossMulti.eval()
    
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
    if args.tripleTask:
      totalClassLoss = 0.
      totalCompLoss = 0.
      totalPurLoss = 0.
      totalCompErrorSqSum = 0.
      totalPurErrorSqSum = 0.
    if args.multiTask:
      totalClassLoss = 0.
      totalCompLoss = 0.
      totalErrorSqSum = 0.
      testCompCorrect = 0
      testCorrect_c0 = 0
      testCorrect_c1 = 0
      testCorrect_c2 = 0
      testCorrect_c3 = 0
      testCorrect_c4 = 0
      total_c0 = 0
      total_c1 = 0
      total_c2 = 0
      total_c3 = 0
      total_c4 = 0
    
    with torch.no_grad():
        
        for batch, (X, y) in enumerate(dataloader):
            if n_batches > 0 and tstep >= n_batches:
                break
            if tstep % args.log_frequency == 0:
                print("reached validation batch %i of %i"%(tstep, testSteps), flush=True)
            if args.tripleTask:
                yComp = y[1]
                yPur = y[2]
                y = y[0].type(torch.LongTensor)
                X, y, yComp, yPur = X.to(args.device), y.to(args.device), yComp.to(args.device), yPur.to(args.device)
                outputs = model(X)
                losses, loss, lossWeights = lossMulti(outputs, [y, yComp, yPur])
                pred = outputs[0].to(args.device)
                pred_comp = outputs[1].to(args.device)
                pred_pur = outputs[2].to(args.device)
                totalTestLoss += loss.detach().item()
                totalClassLoss += losses[0].detach().item()
                totalCompLoss += losses[1].detach().item()
                totalPurLoss += losses[2].detach().item()
                totalCompErrorSqSum += torch.square(torch.sub(yComp, pred_comp)).sum().item()
                totalPurErrorSqSum += torch.square(torch.sub(yPur, pred_pur)).sum().item()
            elif args.multiTask:
                if args.classifyComp:
                    yComp = y[1].type(torch.LongTensor)
                else:
                    yComp = y[1]
                y = y[0].type(torch.LongTensor)
                X, y, yComp = X.to(args.device), y.to(args.device), yComp.to(args.device)
                outputs = model(X)
                losses, loss, lossWeights = lossMulti(outputs, [y, yComp])
                pred = outputs[0].to(args.device)
                pred_comp = outputs[1].to(args.device)
                totalTestLoss += loss.detach().item()
                totalClassLoss += losses[0].detach().item()
                totalCompLoss += losses[1].detach().item()
                if not args.classifyComp:
                    totalErrorSqSum += torch.square(torch.sub(yComp, pred_comp)).sum().item()
            elif args.softLabels:
                target = y
                y = y.argmax(1)
                X, y, target = X.to(args.device), y.to(args.device), target.to(args.device)
                pred = model(X)
                totalTestLoss += softNLLLoss(pred, target)
            else:
                y = y.type(torch.LongTensor)
                X, y = X.to(args.device), y.to(args.device)
                pred = model(X)
                totalTestLoss += lossFn(pred, y).detach().item()
            
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

            if args.multiTask and args.classifyComp:

                iC0 = (yComp == 0).nonzero(as_tuple=True)
                iC1 = (yComp == 1).nonzero(as_tuple=True)
                iC2 = (yComp == 2).nonzero(as_tuple=True)
                iC3 = (yComp == 3).nonzero(as_tuple=True)
                iC4 = (yComp == 4).nonzero(as_tuple=True)
    
                total_c0 += yComp[iC0].size(dim=0)
                total_c1 += yComp[iC1].size(dim=0)
                total_c2 += yComp[iC2].size(dim=0)
                total_c3 += yComp[iC3].size(dim=0)
                total_c4 += yComp[iC4].size(dim=0)
    
                testCompCorrect += (pred_comp.argmax(1) == yComp).type(torch.float).sum().item() 
                if yComp[iC0].size(dim=0) > 0:
                    testCorrect_c0 += (pred_comp[iC0].argmax(1) == yComp[iC0]).type(torch.float).sum().item()
                if yComp[iC1].size(dim=0) > 0:
                    testCorrect_c1 += (pred_comp[iC1].argmax(1) == yComp[iC1]).type(torch.float).sum().item()
                if yComp[iC2].size(dim=0) > 0:
                    testCorrect_c2 += (pred_comp[iC2].argmax(1) == yComp[iC2]).type(torch.float).sum().item()
                if yComp[iC3].size(dim=0) > 0:
                    testCorrect_c3 += (pred_comp[iC3].argmax(1) == yComp[iC3]).type(torch.float).sum().item()
                if yComp[iC4].size(dim=0) > 0:
                    testCorrect_c4 += (pred_comp[iC4].argmax(1) == yComp[iC4]).type(torch.float).sum().item()

            tstep += 1
            gc.collect()
            
    avgTestLoss = totalTestLoss / tstep
    testAcc = testCorrect / (tstep*dataloader.batch_size)
    testAcc_e = testCorrect_e / total_e if (total_e > 0) else -1.
    testAcc_ph = testCorrect_ph / total_ph if (total_ph > 0) else -1.
    testAcc_mu = testCorrect_mu / total_mu if (total_mu > 0) else -1.
    testAcc_pi = testCorrect_pi / total_pi if (total_pi > 0) else -1.
    testAcc_pr = testCorrect_pr / total_pr if (total_pr > 0) else -1.
    testAcc_o = testCorrect_o / total_o if (total_o > 0) else -1.
    if args.tripleTask:
      avgClassLoss = totalClassLoss / tstep
      avgCompLoss = totalCompLoss / tstep
      avgPurLoss = totalPurLoss / tstep
      testCompRMSE = sqrt( totalCompErrorSqSum / (tstep*dataloader.batch_size) )
      testPurRMSE = sqrt( totalPurErrorSqSum / (tstep*dataloader.batch_size) )
    if args.multiTask:
      avgClassLoss = totalClassLoss / tstep
      avgCompLoss = totalCompLoss / tstep
      if args.classifyComp:
        testCompAcc = testCompCorrect / (tstep*dataloader.batch_size)
        testCompAcc_c0 = testCorrect_c0 / total_c0 if (total_c0 > 0) else -1.
        testCompAcc_c1 = testCorrect_c1 / total_c1 if (total_c1 > 0) else -1.
        testCompAcc_c2 = testCorrect_c2 / total_c2 if (total_c2 > 0) else -1.
        testCompAcc_c3 = testCorrect_c3 / total_c3 if (total_c3 > 0) else -1.
        testCompAcc_c4 = testCorrect_c4 / total_c4 if (total_c4 > 0) else -1.
      else:
        testRMSE = sqrt( totalErrorSqSum / (tstep*dataloader.batch_size) )

    if args.tripleTask:
      return avgTestLoss, avgClassLoss, testAcc, testAcc_e, testAcc_ph, testAcc_mu, testAcc_pi, testAcc_pr, testAcc_o, avgCompLoss, testCompRMSE, avgPurLoss, testPurRMSE
    if args.multiTask:
      if args.classifyComp:
        return avgTestLoss, avgClassLoss, testAcc, testAcc_e, testAcc_ph, testAcc_mu, testAcc_pi, testAcc_pr, testAcc_o, avgCompLoss, testCompAcc, testCompAcc_c0, testCompAcc_c1, testCompAcc_c2, testCompAcc_c3, testCompAcc_c4 
      return avgTestLoss, avgClassLoss, testAcc, testAcc_e, testAcc_ph, testAcc_mu, testAcc_pi, testAcc_pr, testAcc_o, avgCompLoss, testRMSE
    return avgTestLoss, testAcc, testAcc_e, testAcc_ph, testAcc_mu, testAcc_pi, testAcc_pr, testAcc_o



def train(train_dataloader, test_dataloader, step, logStep, epoch):

    model.train()
    if args.multiTask or args.tripleTask:
        lossMulti.train()

    totalTrainLoss = 0
    trainCorrect = 0
    trainSteps = len(train_dataloader.dataset) // train_dataloader.batch_size
    dataloading_time = 0.
    backprop_time = 0.
    if args.tripleTask:
      totalClassLoss = 0.
      totalCompLoss = 0.
      totalPurLoss = 0.
      totalCompErrorSqSum = 0.
      totalPurErrorSqSum = 0.
    if args.multiTask:
      totalClassLoss = 0.
      totalCompLoss = 0.
      totalErrorSqSum = 0.
      trainCompCorrect = 0
    
    start = time.time()
    for batch, (X,y) in enumerate(train_dataloader):
        dataloading_time += time.time() - start
        if args.tripleTask:
            yComp = y[1]
            yPur = y[2]
            y = y[0].type(torch.LongTensor)
            X, y, yComp, yPur = X.to(args.device), y.to(args.device), yComp.to(args.device), yPur.to(args.device)
            outputs = model(X)
            losses, loss, lossWeights = lossMulti(outputs, [y, yComp, yPur])
            pred = outputs[0].to(args.device)
            pred_comp = outputs[1].to(args.device)
            pred_pur = outputs[2].to(args.device)
        elif args.multiTask:
            if args.classifyComp:
                yComp = y[1].type(torch.LongTensor)
            else:
                yComp = y[1]
            y = y[0].type(torch.LongTensor)
            X, y, yComp = X.to(args.device), y.to(args.device), yComp.to(args.device)
            outputs = model(X)
            losses, loss, lossWeights = lossMulti(outputs, [y, yComp])
            pred = outputs[0].to(args.device)
            pred_comp = outputs[1].to(args.device)
        elif args.softLabels:
            target = y
            y = y.argmax(1)
            X, y, target = X.to(args.device), y.to(args.device), target.to(args.device)
            pred = model(X)
            loss = softNLLLoss(pred, target)
        else:
            y = y.type(torch.LongTensor)
            X, y = X.to(args.device), y.to(args.device)
            pred = model(X)
            loss = lossFn(pred, y)
        
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
        if args.tripleTask:
            lossClassVal = losses[0].detach().item()
            lossCompVal = losses[1].detach().item()
            lossPurVal = losses[2].detach().item()
            totalClassLoss += lossClassVal
            totalCompLoss += lossCompVal
            totalPurLoss += lossPurVal
            batchCompErrorSqSum = torch.square(torch.sub(yComp, pred_comp)).sum().item()
            totalCompErrorSqSum += batchCompErrorSqSum
            batchCompRMSE = sqrt(batchCompErrorSqSum / train_dataloader.batch_size)
            batchPurErrorSqSum = torch.square(torch.sub(yPur, pred_pur)).sum().item()
            totalPurErrorSqSum += batchPurErrorSqSum
            batchPurRMSE = sqrt(batchPurErrorSqSum / train_dataloader.batch_size)
            lossWClassVal = lossWeights[0].detach().item()
            lossWCompVal = lossWeights[1].detach().item()
            lossWPurVal = lossWeights[2].detach().item()
        if args.multiTask:
            lossClassVal = losses[0].detach().item()
            lossCompVal = losses[1].detach().item()
            if not args.hardWeights:
                lossWClassVal = lossWeights[0].detach().item()
                lossWCompVal = lossWeights[1].detach().item()
            totalClassLoss += lossClassVal
            totalCompLoss += lossCompVal
            if args.classifyComp:
                batchCompCorrect = (pred_comp.argmax(1) == yComp).type(torch.float).sum().item()
                trainCompCorrect += batchCompCorrect
                batchCompAcc = batchCompCorrect / train_dataloader.batch_size
            else:
                batchErrorSqSum = torch.square(torch.sub(yComp, pred_comp)).sum().item()
                totalErrorSqSum += batchErrorSqSum
                batchRMSE = sqrt(batchErrorSqSum / train_dataloader.batch_size)

        if step % args.log_frequency == 0:
            print("reached training batch %i of %i"%(batch, trainSteps), flush=True)
            if args.tripleTask:
              valLoss, valClassLoss, valAcc, valAcc_e, valAcc_ph, valAcc_mu, valAcc_pi, valAcc_pr, valAcc_o, valCompLoss, valCompRMSE, valPurLoss, valPurRMSE = test(test_dataloader, args.n_val_batches)
            elif args.multiTask:
              if args.classifyComp:
                valLoss, valClassLoss, valAcc, valAcc_e, valAcc_ph, valAcc_mu, valAcc_pi, valAcc_pr, valAcc_o, valCompLoss, valCompAcc, valCompAcc_c0, valCompAcc_c1, valCompAcc_c2, valCompAcc_c3, valCompAcc_c4  = test(test_dataloader, args.n_val_batches)
              else:
                valLoss, valClassLoss, valAcc, valAcc_e, valAcc_ph, valAcc_mu, valAcc_pi, valAcc_pr, valAcc_o, valCompLoss, valRMSE = test(test_dataloader, args.n_val_batches)
            else:
                valLoss, valAcc, valAcc_e, valAcc_ph, valAcc_mu, valAcc_pi, valAcc_pr, valAcc_o = test(test_dataloader, args.n_val_batches)
            if not args.noLogs:
              currentLR = args.learning_rate
              if args.schedStepLR or useNonStepScheduler:
                currentLR = optimizer.param_groups[0]["lr"]
              if args.tripleTask:
                wandb.log({"train_loss": lossVal, "train_class_loss": lossClassVal, "train_class_acc": batchAcc,
                           "train_comp_loss": lossCompVal, "train_comp_rmse": batchCompRMSE,
                           "train_purity_loss": lossPurVal, "train_purity_rmse": batchPurRMSE,
                           "val_loss": valLoss, "val_class_loss": valClassLoss, "val_acc": valAcc,
                           "val_comp_loss": valCompLoss, "val_comp_rmse": valCompRMSE,
                           "val_purity_loss": valPurLoss, "val_purity_rmse": valPurRMSE,
                           "val_electron_acc": valAcc_e, "val_photon_acc": valAcc_ph, "val_muon_acc": valAcc_mu,
                           "val_pion_acc": valAcc_pi, "val_proton_acc": valAcc_pr, "class_loss_weight": lossWClassVal,
                           "comp_loss_weight":lossWCompVal, "purity_loss_weight":lossWPurVal,
                           "epoch": epoch, "step": step, "learning_rate": currentLR}, step=logStep)
              elif args.multiTask:
                if args.classifyComp:
                  wandb.log({"train_loss": lossVal, "train_class_loss": lossClassVal, "train_comp_loss": lossCompVal,
                             "train_class_acc": batchAcc, "train_comp_acc": batchCompAcc,
                             "val_loss": valLoss, "val_class_loss": valClassLoss, "val_comp_loss": valCompLoss,
                             "val_acc": valAcc, "val_electron_acc": valAcc_e, "val_photon_acc": valAcc_ph,
                             "val_muon_acc": valAcc_mu, "val_pion_acc": valAcc_pi, "val_proton_acc": valAcc_pr,
                             "val_comp_acc": valCompAcc, "val_c0_acc": valCompAcc_c0, "val_c1_acc": valCompAcc_c1, 
                             "val_c2_acc": valCompAcc_c2, "val_c3_acc": valCompAcc_c3, "val_c4_acc": valCompAcc_c4, 
                             "class_loss_weight": lossWClassVal, "comp_loss_weight":lossWCompVal,
                             "epoch": epoch, "step": step, "learning_rate": currentLR}, step=logStep)
                elif args.hardWeights:
                  wandb.log({"train_loss": lossVal, "train_class_loss": lossClassVal, "train_comp_loss": lossCompVal,
                             "train_class_acc": batchAcc, "train_comp_rmse": batchRMSE,
                             "val_loss": valLoss, "val_class_loss": valClassLoss, "val_comp_loss": valCompLoss,
                             "val_acc": valAcc, "val_comp_rmse": valRMSE,
                             "val_electron_acc": valAcc_e, "val_photon_acc": valAcc_ph, "val_muon_acc": valAcc_mu,
                             "val_pion_acc": valAcc_pi, "val_proton_acc": valAcc_pr,
                             "epoch": epoch, "step": step, "learning_rate": currentLR}, step=logStep)
                else:
                  wandb.log({"train_loss": lossVal, "train_class_loss": lossClassVal, "train_comp_loss": lossCompVal,
                             "train_class_acc": batchAcc, "train_comp_rmse": batchRMSE,
                             "val_loss": valLoss, "val_class_loss": valClassLoss, "val_comp_loss": valCompLoss,
                             "val_acc": valAcc, "val_comp_rmse": valRMSE,
                             "val_electron_acc": valAcc_e, "val_photon_acc": valAcc_ph, "val_muon_acc": valAcc_mu,
                             "val_pion_acc": valAcc_pi, "val_proton_acc": valAcc_pr,
                             "class_loss_weight": lossWClassVal, "comp_loss_weight":lossWCompVal,
                             "epoch": epoch, "step": step, "learning_rate": currentLR}, step=logStep)
              elif args.use6class:
                  wandb.log({"train_loss": lossVal, "train_acc": batchAcc, "val_loss": valLoss, "val_acc": valAcc,
                             "val_electron_acc": valAcc_e, "val_photon_acc": valAcc_ph, "val_muon_acc": valAcc_mu,
                             "val_pion_acc": valAcc_pi, "val_proton_acc": valAcc_pr, "val_other_acc": valAcc_o, 
                             "epoch": epoch, "step": step, "learning_rate": currentLR}, step=logStep)
              else:
                  wandb.log({"train_loss": lossVal, "train_acc": batchAcc, "val_loss": valLoss, "val_acc": valAcc,
                             "val_electron_acc": valAcc_e, "val_photon_acc": valAcc_ph, "val_muon_acc": valAcc_mu,
                             "val_pion_acc": valAcc_pi, "val_proton_acc": valAcc_pr,
                             "epoch": epoch, "step": step, "learning_rate": currentLR}, step=logStep)
            logStep += 1
            model.train()
            if args.multiTask or args.tripleTask:
                lossMulti.train()

        step += 1
        if useNonStepScheduler:
          scheduler.step()
        gc.collect()
        start = time.time()
        
    avgTrainLoss = totalTrainLoss / trainSteps
    trainAcc = trainCorrect / len(train_dataloader.dataset)
    if args.tripleTask:
      avgClassLoss = totalClassLoss /  trainSteps
      avgCompLoss = totalCompLoss / trainSteps
      avgPurLoss = totalPurLoss / trainSteps
      trainCompRMSE = sqrt( totalCompErrorSqSum / len(train_dataloader.dataset) )
      trainPurRMSE = sqrt( totalPurErrorSqSum / len(train_dataloader.dataset) )
    if args.multiTask:
      avgClassLoss = totalClassLoss /  trainSteps
      avgCompLoss = totalCompLoss / trainSteps
      if args.classifyComp:
        trainCompAcc = trainCompCorrect / len(train_dataloader.dataset)
      else:
        trainRMSE = sqrt( totalErrorSqSum / len(train_dataloader.dataset) )
    
    print("total time spent loading data:   ", dataloading_time, flush=True)
    print("total time spent doing backprop: ", backprop_time, flush=True)

    if args.tripleTask:
      return step, logStep, avgTrainLoss, avgClassLoss, trainAcc, avgCompLoss, trainCompRMSE, avgPurLoss, trainPurRMSE
    if args.multiTask:
      if args.classifyComp:
        return step, logStep, avgTrainLoss, avgClassLoss, trainAcc, avgCompLoss, trainCompAcc
      return step, logStep, avgTrainLoss, avgClassLoss, trainAcc, avgCompLoss, trainRMSE
    return step, logStep, avgTrainLoss, trainAcc




for e in range(args.startEpoch, args.epochs+args.startEpoch):
  if args.cyclicLRMode in ["log_triangular","log_triangular2"]:
    if (e - 1) % args.schedLRStepSize == 0:
      ascending = not ascending
      if e > 1 and ascending and args.cyclicLRMode == "log_triangular2":
        schedLRMax = (args.schedLRBase/args.schedLRMax)**(1./(nCycles))*schedLRMax
      optimizer.param_groups[0]["lr"] = lambda1(step)*args.learning_rate #override bug when switching directions
  if args.tripleTask:
    step, logStep, trL, trClL, trA, trCoL, trCoRMSE, trPuL, trPuRMSE = train(train_dataloader, test_dataloader, step, logStep, e)
    teL, teClL, teA, teA_e, teA_ph, teA_mu, teA_pi, teA_pr, teA_o, teCoL, teCoRMSE, tePuL, tePuRMSE = test(test_dataloader)
    print("EPOCH:", e, " train total loss:", trL, " train class loss:", trClL, " train class accuracy:", trA,
          " train comp loss:", trCoL, " train comp RMSE:", trCoRMSE,
          " train purity loss:", trPuL, " train purity RMSE:", trPuRMSE,
          " test total loss:", teL, " test class loss:", teClL, " test class accuracy:", teA,
          " test comp loss:", teCoL, " test comp RMSE:", teCoRMSE,
          " test purity loss:", tePuL, " test purity RMSE:", tePuRMSE,
          " electron test accuracy:", teA_e, " photon test accuracy:", teA_ph, " muon test accuracy:", teA_mu,
          " pion test accuracy:", teA_pi, " proton test accuracy:", teA_pr, " other test accuracy:", teA_o,
          flush=True)
  elif args.multiTask:
    if args.classifyComp:
      step, logStep, trL, trClL, trA, trCoL, trCmpA = train(train_dataloader, test_dataloader, step, logStep, e)
      teL, teClL, teA, teA_e, teA_ph, teA_mu, teA_pi, teA_pr, teA_o, teCoL, teCoA, teCoA0, teCoA1, teCoA2, teCoA3, teCoA4 = test(test_dataloader)
      print("EPOCH:", e, " train total loss:", trL, " train class loss:", trClL, " train comp loss:", trCoL,
            " train class accuracy:", trA, " train comp accuracy:", trCmpA,
            " test total loss:", teL, " test class loss:", teClL, "test comp loss:", teCoL,
            " test class accuracy:", teA, " electron test accuracy:", teA_e, " photon test accuracy:", teA_ph,
            " muon test accuracy:", teA_mu, " pion test accuracy:", teA_pi, " proton test accuracy:", teA_pr,
            " other test accuracy:", teA_o, "test comp accuracy:", teCoA, "test comp class0 accuracy:", teCoA0,
            "test comp class1 accuracy:", teCoA1, "test comp class2 accuracy:", teCoA2,
            "test comp class3 accuracy:", teCoA3, "test comp class4 accuracy:", teCoA4, flush=True)
    else:
      step, logStep, trL, trClL, trA, trCoL, trRMSE = train(train_dataloader, test_dataloader, step, logStep, e)
      teL, teClL, teA, teA_e, teA_ph, teA_mu, teA_pi, teA_pr, teA_o, teCoL, teRMSE = test(test_dataloader)
      print("EPOCH:", e, " train total loss:", trL, " train class loss:", trClL, " train comp loss:", trCoL,
            " train class accuracy:", trA, " train comp RMSE:", trRMSE,
            " test total loss:", teL, " test class loss:", teClL, "test comp loss:", teCoL,
            " test class accuracy:", teA, " test comp RMSE:", teRMSE,
            " electron test accuracy:", teA_e, " photon test accuracy:", teA_ph, " muon test accuracy:", teA_mu,
            " pion test accuracy:", teA_pi, " proton test accuracy:", teA_pr, " other test accuracy:", teA_o,
            flush=True)
  else:
    step, logStep, trL, trA = train(train_dataloader, test_dataloader, step, logStep, e)
    teL, teA, teA_e, teA_ph, teA_mu, teA_pi, teA_pr, teA_o = test(test_dataloader)
    print("EPOCH:", e, " train loss:",trL, " train accuracy:", trA, " test loss:", teL, " test accuracy:", teA,
          " electron test accuracy:", teA_e, " photon test accuracy:", teA_ph, " muon test accuracy:", teA_mu,
          " pion test accuracy:", teA_pi, " proton test accuracy:", teA_pr, " other test accuracy:", teA_o, flush=True)
  if args.multiTask:
    weight_state_dict = {'etaC': lossMulti.etaC.detach().item(),
                         'etaR': lossMulti.etaR.detach().item()}
    torch.save({'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'weight_state_dict': weight_state_dict},
               args.model_path.replace(".pt", "_epoch%i.pt"%e))
  elif args.tripleTask:
    weight_state_dict = {'etaC': lossMulti.etaC.detach().item(),
                         'etaRc': lossMulti.etaRc.detach().item(),
                         'etaRp': lossMulti.etaRp.detach().item()}
    torch.save({'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'weight_state_dict': weight_state_dict},
               args.model_path.replace(".pt", "_epoch%i.pt"%e))
  else:
    torch.save({'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()},
               args.model_path.replace(".pt", "_epoch%i.pt"%e))
  if args.schedStepLR:
    scheduler.step()

print("last training step: ", step)
print("last log step: ", logStep)

