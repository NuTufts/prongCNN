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
from torch.optim.lr_scheduler import OneCycleLR

import matplotlib.pyplot as plt 

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/models')

from datasets_reco_5ClassHardLabel_quadTask import ProngDataset, mean, std
from models_instanceNorm_reco_2chan_quadTask import ResBlock, ResNet34


parser = argparse.ArgumentParser("kludge script to generate checkpoint to resume one-cycle cosine annealing learning rate scheduler")
parser.add_argument("-t", "--train_file", type=str, default="images/prongCNN_reco_v2me05_images_file_run3bNuAndNueOverlays_altCVnu_preprocess_v03_cleaned_minHit10_noSecondaries_noPurityCut_2000PerClassVal_train.root")
parser.add_argument("-l", "--learning_rate", type=float, default=1e-8, help="learning rate (initial for OneCycle)")
parser.add_argument("-lrM", "--schedLRMax", type=float, default=1e-2, help="maximum learning rate for oscillatory and one cycle LR schedulers")
parser.add_argument("-e", "--epochs", type=int, default=20, help="number of training epochs")
parser.add_argument("-cE", "--checkpointEpoch", type=int, default=16, help="write scheduler checkpoint for this epoch")
parser.add_argument("-bt", "--batch_size_train", type=int, default=64, help="training batch size")
parser.add_argument("--printCurve", action="store_true", help="write lr vs step arrays to file")
parser.add_argument("--saveOptimizer", action="store_true", help="write lr vs step arrays to file")
args = parser.parse_args()

train_dataset = ProngDataset(args.train_file, clip=4.0)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, drop_last=True, shuffle=True)

itersPerEpoch = len(train_dataloader.dataset) // train_dataloader.batch_size

model = ResNet34(2, ResBlock, outputs=5)
model.to("cuda")

class QuadTaskLoss(nn.Module):
  def __init__(self, wC=0.5, wRc=0.5, wRp=0.5, wCP=0.5):
    super(QuadTaskLoss, self).__init__()
    self.etaC = nn.Parameter(torch.Tensor([wC]))
    self.etaRc = nn.Parameter(torch.Tensor([wRc]))
    self.etaRp = nn.Parameter(torch.Tensor([wRp]))
    self.etaCP = nn.Parameter(torch.Tensor([wCP]))
  def forward(self, outputs, targets):
    loss_class = lossFn(outputs[0], targets[0])
    loss_comp = lossMSEcomp(outputs[1], targets[1])
    loss_pur = lossMSEpur(outputs[2], targets[2])
    loss_classProc = lossFnProc(outputs[3], targets[3])
    loss_total = 2.0*torch.exp(-self.etaC)*loss_class + torch.exp(-self.etaRc)*loss_comp + torch.exp(-self.etaRp)*loss_pur + 2.0*torch.exp(-self.etaCP)*loss_classProc + self.etaC + self.etaRc + self.etaRp + self.etaCP
    return [loss_class, loss_comp, loss_pur, loss_classProc], loss_total, [self.etaC, self.etaRc, self.etaRp, self.etaCP]

initialLossWeights = [0.5, 0.5, 0.5, 0.5]
lossMulti = QuadTaskLoss(initialLossWeights[0], initialLossWeights[1], initialLossWeights[2], initialLossWeights[3]).to("cuda")

optimizer = AdamW(list(lossMulti.parameters())+list(model.parameters()), lr=args.learning_rate)

scheduler = OneCycleLR(optimizer, max_lr=args.schedLRMax, steps_per_epoch=itersPerEpoch, epochs=args.epochs, anneal_strategy='cos')

def advanceSchedulerEpoch(step):
  for batch in range(itersPerEpoch):
    optimizer.step()
    scheduler.step()
    step += 1
    if args.printCurve:
      step_list.write(", %i"%step)
      lr_list.write(", %f"%optimizer.param_groups[0]["lr"])
  return step

step = 0

if args.printCurve:
  step_list = open("make_scheduler_checkpoint_step_array.txt","w")
  step_list.write("steps = [%i"%step)
  lr_list = open("make_scheduler_checkpoint_lr_array.txt","w")
  lr_list.write("lr_list = [%f"%optimizer.param_groups[0]["lr"])

for e in range(1, args.epochs+1):
  print("starting epoch %i"%e)
  step = advanceSchedulerEpoch(step)
  if e == args.checkpointEpoch:
    torch.save(scheduler.state_dict(), "scheduler_state_dict_epoch%i.pt"%e)
    if args.saveOptimizer:
      torch.save(optimizer.state_dict(), "optimizer_state_dict_epoch%i.pt"%e)
    print("final step:", step)
    print("final LR:", optimizer.param_groups[0]["lr"])
    break

if args.printCurve:
  step_list.write("]")
  lr_list.write("]")


