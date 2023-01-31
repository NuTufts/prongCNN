
import argparse
import sys
import os

from larlite import larlite
from larlite import larutil
from ublarcvapp import ublarcvapp
from larcv import larcv
from larflow import larflow

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/models')


parser = argparse.ArgumentParser("evaluate Prong CNN")
parser.add_argument("-i", "--images_file", type=str, required=True, help="validation images file")
parser.add_argument("-m", "--model_path", type=str, required=True, help="model name")
parser.add_argument("-d", "--device", type=str, default="cuda", help="gpu/cpu device")
parser.add_argument("-n", "--num_workers", type=int, default=12, help="number of cpu workers for data loading")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="validation batch size")
parser.add_argument("-c", "--l0inChans", type=int, default=2, help="number of input channels for first conv layer")
parser.add_argument("--multiTask", action="store_true", help="do particle classification and completeness regression")
parser.add_argument("--classifyComp", action="store_true", help="do classification instead of regression for completeness")
parser.add_argument("--use6class", action="store_true", help="use 6 classes (include other label)")
parser.add_argument("--softLabels", action="store_true", help="use soft labels for loss")
parser.add_argument("--noMask", action="store_true", help="only use prong pixels")
parser.add_argument("--plane2only", action="store_true", help="only use collection plane images")
parser.add_argument("--resnet18", action="store_true", help="use ResNet18 instead of ResNet34")
parser.add_argument("--singleGPU", action="store_true", help="only use one GPU")
args = parser.parse_args()

if args.multiTask and (args.l0inChans != 2 or args.use6class or args.softLabels or args.noMask or args.plane2only or args.resnet18):
  sys.exit("multiTask training only configured for 5 class hard labels with mask (3 plane, 2 in channel config.) with ResNet34")

if args.multiTask:
  from models_instanceNorm_reco_2chan_multiTask import ResBlock, ResNet34, ResNet34ClCmp
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
if args.multiTask:
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


if args.plane2only:
    img_mean = meanPl2
    img_std = stdPl2
    if args.noMask:
      img_mean = meanPl2_nm
      img_std = stdPl2_nm
    transform = transforms.Normalize(img_mean, img_std)

    if args.noMask:
      dataset = ProngDatasetPl2NoMask(args.images_file, transformations=transform, clip=4.0)
    else:
      dataset = ProngDatasetPl2(args.images_file, transformations=transform, clip=4.0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=args.num_workers)

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
    transform = transforms.Normalize(img_mean, img_std)

    if args.noMask:
      dataset = ProngDatasetNoMask(args.images_file, transformations=transform, clip=4.0)
    else:
      if args.multiTask and args.classifyComp:
        dataset = ProngDatasetClCmp(args.images_file, transformations=transform, clip=4.0)
      else:
        dataset = ProngDataset(args.images_file, transformations=transform, clip=4.0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=args.num_workers)

    if args.resnet18:
        model = ResNet18(layer0inChans, ResBlock, outputs=nClasses)
    else:
        model = ResNet34(layer0inChans, ResBlock, outputs=nClasses)
    if not args.singleGPU:
        model = nn.DataParallel(model)


model.load_state_dict(torch.load(args.model_path))
model.to(args.device)
model.eval()

#print(model)


class classCounter:
    def __init__(self):
        self.n = 0
        self.n_match = [0,0,0,0,0,0]
    def update(self, match):
        self.n += 1
        self.n_match[match] += 1


def test(dataloader, model):
    
    model.eval()
    
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
    
    effCounts = {0: classCounter(), 1: classCounter(), 2: classCounter(), 3: classCounter(), 4: classCounter(), 5: classCounter()} 
    purCounts = {0: classCounter(), 1: classCounter(), 2: classCounter(), 3: classCounter(), 4: classCounter(), 5: classCounter()} 

    with torch.no_grad():
        
        for batch, (X, y) in enumerate(dataloader):
            if tstep % 1000 == 0:
                print("reached validation batch %i of %i"%(tstep, testSteps), flush=True)
            if args.multiTask:
                if args.classifyComp:
                    yComp = y[1].type(torch.LongTensor)
                else:
                    yComp = y[1]
                    #yCompCl = []
                y = y[0].type(torch.LongTensor)
                X, y, yComp = X.to(args.device), y.to(args.device), yComp.to(args.device)
                outputs = model(X)
                pred = outputs[0]
                pred_comp = outputs[1]
            elif args.softLabels:
                y = y.argmax(1)
                X, y = X.to(args.device), y.to(args.device)
                pred = model(X)
            else:
                y = y.type(torch.LongTensor)
                X, y = X.to(args.device), y.to(args.device)
                pred = model(X)
            y_pred = pred.argmax(1)
            
            for i in range(y.size(0)):
              effCounts[y[i].item()].update(y_pred[i].item())
              purCounts[y_pred[i].item()].update(y[i].item())
            
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

            tstep += 1
            
    testAcc = testCorrect / len(dataloader.dataset)
    testAcc_e = testCorrect_e / total_e
    testAcc_ph = testCorrect_ph / total_ph
    testAcc_mu = testCorrect_mu / total_mu
    testAcc_pi = testCorrect_pi / total_pi
    testAcc_pr = testCorrect_pr / total_pr
    testAcc_o = 0.
    if total_o > 0:
        testAcc_o = testCorrect_o / total_o
    
    return testAcc, testAcc_e, testAcc_ph, testAcc_mu, testAcc_pi, testAcc_pr, testAcc_o, effCounts, purCounts





teA, teA_e, teA_ph, teA_mu, teA_pi, teA_pr, teA_o, effCounts, purCounts = test(dataloader, model)
print("test accuracy:", teA, " electron test accuracy:", teA_e, " photon test accuracy:", teA_ph, " muon test accuracy:", teA_mu, " pion test accuracy:", teA_pi, " proton test accuracy:", teA_pr, " other test accuracy:", teA_o, flush=True)

for i in range(6):
    print("true class %i matches (%i instances):"%(i, effCounts[i].n))
    for j in range(5):
        ratio = 0.
        if effCounts[i].n > 0:
            ratio = effCounts[i].n_match[j]/effCounts[i].n
        print("    %i: %f"%(j, ratio))
for i in range(6):
    print("pred class %i matches (%i instances):"%(i, purCounts[i].n))
    for j in range(5):
        ratio = 0.
        if purCounts[i].n > 0:
            ratio = purCounts[i].n_match[j]/purCounts[i].n
        print("    %i: %f"%(j, ratio))


