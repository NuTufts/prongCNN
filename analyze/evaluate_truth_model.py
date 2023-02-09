
import argparse
import sys
import os

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/models')

from datasets import ProngDataset, ProngDatasetPl2, mean, std, meanPl2, stdPl2
#from models import ResBlock, ResNet18, ResNet18Pl2, ResNet34, ResNet34Pl2
from models_instanceNorm import ResBlock, ResNet18, ResNet18Pl2, ResNet34, ResNet34Pl2


parser = argparse.ArgumentParser("evaluate Prong CNN")
parser.add_argument("-i", "--images_file", type=str, default="images/prongCNN_images_file_5particle_test.root", help="validation images file")
parser.add_argument("-d", "--device", type=str, default="cuda", help="gpu/cpu device")
parser.add_argument("-n", "--num_workers", type=int, default=12, help="number of cpu workers for data loading")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="validation batch size")
parser.add_argument("-m", "--model_path", type=str, default="models/ResNet34_5part_allPls_pixClip4_instNorm_biasInNorm/ResNet34_5part_plAll_epoch10.pt", help="model name")
parser.add_argument("--plane2only", action="store_true", help="only use collection plane images")
parser.add_argument("--resnet18", action="store_true", help="use ResNet18 instead of ResNet34")
parser.add_argument("--singleGPU", action="store_true", help="only use one GPU")
args = parser.parse_args()

if args.plane2only:
    transform = transforms.Normalize(meanPl2, stdPl2)
    dataset = ProngDatasetPl2(args.images_file, transformations=transform, clip=4.0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=args.num_workers)

    if args.resnet18:
        model = ResNet18Pl2(1, ResBlock, outputs=5, bnRunStat=False)
    else:
        model = ResNet34Pl2(1, ResBlock, outputs=5, bnRunStat=False)
    if not args.singleGPU:
        model = nn.DataParallel(model)

else:
    transform = transforms.Normalize(mean, std)
    dataset = ProngDataset(args.images_file, transformations=transform, clip=4.0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=args.num_workers)

    if args.resnet18:
        model = ResNet18(1, ResBlock, outputs=5, bnRunStat=False)
    else:
        model = ResNet34(1, ResBlock, outputs=5, bnRunStat=False)
    if not args.singleGPU:
        model = nn.DataParallel(model)

model.load_state_dict(torch.load(args.model_path))
model.to(args.device)
model.eval()

#print(model)


class classCounter:
    def __init__(self):
        self.n = 0
        self.n_match = [0,0,0,0,0]
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
    total_e = 0
    total_ph = 0
    total_mu = 0
    total_pi = 0
    total_pr = 0
    testSteps = len(dataloader.dataset) // dataloader.batch_size
    tstep = 0
    
    effCounts = {0: classCounter(), 1: classCounter(), 2: classCounter(), 3: classCounter(), 4: classCounter()} 
    purCounts = {0: classCounter(), 1: classCounter(), 2: classCounter(), 3: classCounter(), 4: classCounter()} 

    with torch.no_grad():
        
        for batch, (X, y) in enumerate(dataloader):
            if tstep % 1000 == 0:
                print("reached validation batch %i of %i"%(tstep, testSteps), flush=True)
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

            total_e += y[iEl].size(dim=0)
            total_ph += y[iPh].size(dim=0)
            total_mu += y[iMu].size(dim=0)
            total_pi += y[iPi].size(dim=0)
            total_pr += y[iPr].size(dim=0)

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

            tstep += 1
            
    testAcc = testCorrect / len(dataloader.dataset)
    testAcc_e = testCorrect_e / total_e
    testAcc_ph = testCorrect_ph / total_ph
    testAcc_mu = testCorrect_mu / total_mu
    testAcc_pi = testCorrect_pi / total_pi
    testAcc_pr = testCorrect_pr / total_pr
    
    return testAcc, testAcc_e, testAcc_ph, testAcc_mu, testAcc_pi, testAcc_pr, effCounts, purCounts



teA, teA_e, teA_ph, teA_mu, teA_pi, teA_pr, effCounts, purCounts = test(dataloader, model)
print("test accuracy:", teA, " electron test accuracy:", teA_e, " photon test accuracy:", teA_ph, " muon test accuracy:", teA_mu, " pion test accuracy:", teA_pi, " proton test accuracy:", teA_pr, flush=True)

for i in range(5):
    print("true class %i matches (%i instances):"%(i, effCounts[i].n))
    for j in range(5):
        print("    %i: %f"%(j, effCounts[i].n_match[j]/effCounts[i].n))
for i in range(5):
    print("pred class %i matches (%i instances):"%(i, purCounts[i].n))
    for j in range(5):
        print("    %i: %f"%(j, purCounts[i].n_match[j]/purCounts[i].n))


