
import argparse
import sys
import os

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import ROOT as rt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/models')


parser = argparse.ArgumentParser("evaluate Prong CNN")
parser.add_argument("-i", "--images_file", type=str, required=True, help="validation images file")
parser.add_argument("-m", "--model_path", type=str, required=True, help="model name")
parser.add_argument("-d", "--device", type=str, default="cuda", help="gpu/cpu device")
parser.add_argument("-n", "--num_workers", type=int, default=12, help="number of cpu workers for data loading")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="validation batch size")
parser.add_argument("-c", "--l0inChans", type=int, default=2, help="number of input channels for first conv layer")
parser.add_argument("-o", "--outfile", type=str, default="evaluate_reco_model_output_plots.root", help="name of output root file with completeness and purity histograms")
parser.add_argument("--multiTask", action="store_true", help="do particle classification and completeness regression")
parser.add_argument("--tripleTask", action="store_true", help="do particle classification and completeness and purity regression")
parser.add_argument("--quadTask", action="store_true", help="do particle and process classification and purity and completeness regression")
parser.add_argument("--classifyComp", action="store_true", help="do classification instead of regression for completeness")
parser.add_argument("--use6class", action="store_true", help="use 6 classes (include other label)")
parser.add_argument("--softLabels", action="store_true", help="use soft labels for loss")
parser.add_argument("--noMask", action="store_true", help="only use prong pixels")
parser.add_argument("--plane2only", action="store_true", help="only use collection plane images")
parser.add_argument("--resnet18", action="store_true", help="use ResNet18 instead of ResNet34")
parser.add_argument("--singleGPU", action="store_true", help="only use one GPU")
parser.add_argument("--noLowPurBins", action="store_true", help="input file has no true purity < 0.6 events")
args = parser.parse_args()

rt.TH1.SetDefaultSumw2(rt.kTRUE)
rt.gStyle.SetOptStat(0)

if args.multiTask and (args.l0inChans != 2 or args.use6class or args.softLabels or args.noMask or args.plane2only or args.resnet18):
  sys.exit("multiTask training only configured for 5 class hard labels with mask (3 plane, 2 in channel config.) with ResNet34")

if args.tripleTask and (args.l0inChans != 2 or args.use6class or args.softLabels or args.noMask or args.plane2only or args.resnet18 or args.multiTask or args.quadTask):
  sys.exit("tripleTask training only configured for 5 class hard labels with mask (3 plane, 2 in channel config.) with ResNet34 and learnable loss weights")

if args.quadTask and (args.l0inChans != 2 or args.use6class or args.softLabels or args.noMask or args.plane2only or args.resnet18 or args.multiTask or args.tripleTask):
  sys.exit("quadTask training only configured for 5 class hard labels with mask (3 plane, 2 in channel config.) with ResNet34 and learnable loss weights")

if args.quadTask:
  from models_instanceNorm_reco_2chan_quadTask import ResBlock, ResNet34
elif args.tripleTask:
  from models_instanceNorm_reco_2chan_tripleTask import ResBlock, ResNet34
elif args.multiTask:
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
nProcClasses = 3
if args.quadTask:
  from datasets_reco_5ClassHardLabel_quadTask import ProngDataset, mean, std
  from datasets_reco_5ClassHardLabel_multiTask import getCompClass
elif args.tripleTask:
  from datasets_reco_5ClassHardLabel_tripleTask import ProngDataset, mean, std
  from datasets_reco_5ClassHardLabel_multiTask import getCompClass
elif args.multiTask:
    from datasets_reco_5ClassHardLabel_multiTask import ProngDataset, ProngDatasetClCmp, mean, std, getCompClass
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
    if "cuda" in args.device and not args.singleGPU:
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
        if args.multiTask and args.classifyComp:
            model = ResNet34ClCmp(layer0inChans, ResBlock, outputs=nClasses)
        else:
            model = ResNet34(layer0inChans, ResBlock, outputs=nClasses)
    if "cuda" in args.device and not args.singleGPU:
        model = nn.DataParallel(model)


#model.load_state_dict(torch.load(args.model_path))
checkpoint = torch.load(args.model_path)
try:
  model.load_state_dict(checkpoint['model_state_dict'])
except:
  model.module.load_state_dict(checkpoint['model_state_dict'])
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

class fiveClassCounter:
    def __init__(self):
        self.n = 0
        self.n_match = [0,0,0,0,0]
    def update(self, match):
        self.n += 1
        self.n_match[match] += 1

class threeClassCounter:
    def __init__(self):
        self.n = 0
        self.n_match = [0,0,0]
    def update(self, match):
        self.n += 1
        self.n_match[match] += 1


if args.multiTask or args.tripleTask or args.quadTask:

  h_comp_predBin0 = rt.TH1F("h_comp_predBin0","True Completeness Distributions, Validation Sample",51,0,1.02)
  h_comp_predBin0.SetLineWidth(2)
  h_comp_predBin0.SetLineColor(rt.kBlack)
  h_comp_predBin0.GetXaxis().SetTitle("true completeness")
  h_comp_predBin1 = rt.TH1F("h_comp_predBin1","True Completeness Distributions, Validation Sample",51,0,1.02)
  h_comp_predBin1.SetLineWidth(2)
  h_comp_predBin1.SetLineColor(40)
  h_comp_predBin1.GetXaxis().SetTitle("true completeness")
  h_comp_predBin2 = rt.TH1F("h_comp_predBin2","True Completeness Distributions, Validation Sample",51,0,1.02)
  h_comp_predBin2.SetLineWidth(2)
  h_comp_predBin2.SetLineColor(8)
  h_comp_predBin2.GetXaxis().SetTitle("true completeness")
  h_comp_predBin3 = rt.TH1F("h_comp_predBin3","True Completeness Distributions, Validation Sample",51,0,1.02)
  h_comp_predBin3.SetLineWidth(2)
  h_comp_predBin3.SetLineColor(rt.kBlue)
  h_comp_predBin3.GetXaxis().SetTitle("true completeness")
  h_comp_predBin4 = rt.TH1F("h_comp_predBin4","True Completeness Distributions, Validation Sample",51,0,1.02)
  h_comp_predBin4.SetLineWidth(2)
  h_comp_predBin4.SetLineColor(rt.kRed)
  h_comp_predBin4.GetXaxis().SetTitle("true completeness")
  
  h_comp_trueBin0 = rt.TH1F("h_comp_trueBin0","Predicted Completeness Distributions, Validation Sample",51,0,1.02)
  h_comp_trueBin0.SetLineWidth(2)
  h_comp_trueBin0.SetLineColor(rt.kBlack)
  h_comp_trueBin0.GetXaxis().SetTitle("predicted completeness")
  h_comp_trueBin1 = rt.TH1F("h_comp_trueBin1","Predicted Completeness Distributions, Validation Sample",51,0,1.02)
  h_comp_trueBin1.SetLineWidth(2)
  h_comp_trueBin1.SetLineColor(40)
  h_comp_trueBin1.GetXaxis().SetTitle("predicted completeness")
  h_comp_trueBin2 = rt.TH1F("h_comp_trueBin2","Predicted Completeness Distributions, Validation Sample",51,0,1.02)
  h_comp_trueBin2.SetLineWidth(2)
  h_comp_trueBin2.SetLineColor(8)
  h_comp_trueBin2.GetXaxis().SetTitle("predicted completeness")
  h_comp_trueBin3 = rt.TH1F("h_comp_trueBin3","Predicted Completeness Distributions, Validation Sample",51,0,1.02)
  h_comp_trueBin3.SetLineWidth(2)
  h_comp_trueBin3.SetLineColor(rt.kBlue)
  h_comp_trueBin3.GetXaxis().SetTitle("predicted completeness")
  h_comp_trueBin4 = rt.TH1F("h_comp_trueBin4","Predicted Completeness Distributions, Validation Sample",51,0,1.02)
  h_comp_trueBin4.SetLineWidth(2)
  h_comp_trueBin4.SetLineColor(rt.kRed)
  h_comp_trueBin4.GetXaxis().SetTitle("predicted completeness")

  h_comp_heatmap = rt.TH2F("h_comp_heatmap","Predicted vs. True Completeness, Validation Sample",26,0,1.04,26,0,1.04)
  h_comp_heatmap.GetXaxis().SetTitle("true completeness")
  h_comp_heatmap.GetYaxis().SetTitle("predicted completeness")
  
  if args.tripleTask or args.quadTask:

    h_pur_predBin0 = rt.TH1F("h_pur_predBin0","True Purity Distributions, Validation Sample",51,0,1.02)
    h_pur_predBin0.SetLineWidth(2)
    h_pur_predBin0.SetLineColor(rt.kBlack)
    h_pur_predBin0.GetXaxis().SetTitle("true purity")
    h_pur_predBin1 = rt.TH1F("h_pur_predBin1","True Purity Distributions, Validation Sample",51,0,1.02)
    h_pur_predBin1.SetLineWidth(2)
    h_pur_predBin1.SetLineColor(40)
    h_pur_predBin1.GetXaxis().SetTitle("true purity")
    h_pur_predBin2 = rt.TH1F("h_pur_predBin2","True Purity Distributions, Validation Sample",51,0,1.02)
    h_pur_predBin2.SetLineWidth(2)
    h_pur_predBin2.SetLineColor(8)
    h_pur_predBin2.GetXaxis().SetTitle("true purity")
    h_pur_predBin3 = rt.TH1F("h_pur_predBin3","True Purity Distributions, Validation Sample",51,0,1.02)
    h_pur_predBin3.SetLineWidth(2)
    h_pur_predBin3.SetLineColor(rt.kBlue)
    h_pur_predBin3.GetXaxis().SetTitle("true purity")
    h_pur_predBin4 = rt.TH1F("h_pur_predBin4","True Purity Distributions, Validation Sample",51,0,1.02)
    h_pur_predBin4.SetLineWidth(2)
    h_pur_predBin4.SetLineColor(rt.kRed)
    h_pur_predBin4.GetXaxis().SetTitle("true purity")
    
    h_pur_trueBin0 = rt.TH1F("h_pur_trueBin0","Predicted Purity Distributions, Validation Sample",51,0,1.02)
    h_pur_trueBin0.SetLineWidth(2)
    h_pur_trueBin0.SetLineColor(rt.kBlack)
    h_pur_trueBin0.GetXaxis().SetTitle("predicted purity")
    h_pur_trueBin1 = rt.TH1F("h_pur_trueBin1","Predicted Purity Distributions, Validation Sample",51,0,1.02)
    h_pur_trueBin1.SetLineWidth(2)
    h_pur_trueBin1.SetLineColor(40)
    h_pur_trueBin1.GetXaxis().SetTitle("predicted purity")
    h_pur_trueBin2 = rt.TH1F("h_pur_trueBin2","Predicted Purity Distributions, Validation Sample",51,0,1.02)
    h_pur_trueBin2.SetLineWidth(2)
    h_pur_trueBin2.SetLineColor(8)
    h_pur_trueBin2.GetXaxis().SetTitle("predicted purity")
    h_pur_trueBin3 = rt.TH1F("h_pur_trueBin3","Predicted Purity Distributions, Validation Sample",51,0,1.02)
    h_pur_trueBin3.SetLineWidth(2)
    h_pur_trueBin3.SetLineColor(rt.kBlue)
    h_pur_trueBin3.GetXaxis().SetTitle("predicted purity")
    h_pur_trueBin4 = rt.TH1F("h_pur_trueBin4","Predicted Purity Distributions, Validation Sample",51,0,1.02)
    h_pur_trueBin4.SetLineWidth(2)
    h_pur_trueBin4.SetLineColor(rt.kRed)
    h_pur_trueBin4.GetXaxis().SetTitle("predicted purity")

    h_pur_heatmap = rt.TH2F("h_pur_heatmap","Predicted vs. True Purity, Validation Sample",26,0,1.04,26,0,1.04)
    h_pur_heatmap.GetXaxis().SetTitle("true purity")
    h_pur_heatmap.GetYaxis().SetTitle("predicted purity")


def fillCompHistos(trueVal, predVal, trueBin, predBin):

  h_comp_heatmap.Fill(trueVal, predVal)

  if predVal < 0.2:
    h_comp_predBin0.Fill(trueVal)
  elif predVal < 0.4:
    h_comp_predBin1.Fill(trueVal)
  elif predVal < 0.6:
    h_comp_predBin2.Fill(trueVal)
  elif predVal < 0.8:
    h_comp_predBin3.Fill(trueVal)
  else:
    h_comp_predBin4.Fill(trueVal)

  if trueVal < 0.2:
    h_comp_trueBin0.Fill(predVal)
  elif trueVal < 0.4:
    h_comp_trueBin1.Fill(predVal)
  elif trueVal < 0.6:
    h_comp_trueBin2.Fill(predVal)
  elif trueVal < 0.8:
    h_comp_trueBin3.Fill(predVal)
  else:
    h_comp_trueBin4.Fill(predVal)


def fillPurityHistos(trueVal, predVal, trueBin, predBin):

  h_pur_heatmap.Fill(trueVal, predVal)

  if predVal < 0.2:
    h_pur_predBin0.Fill(trueVal)
  elif predVal < 0.4:
    h_pur_predBin1.Fill(trueVal)
  elif predVal < 0.6:
    h_pur_predBin2.Fill(trueVal)
  elif predVal < 0.8:
    h_pur_predBin3.Fill(trueVal)
  else:
    h_pur_predBin4.Fill(trueVal)

  if trueVal < 0.2:
    h_pur_trueBin0.Fill(predVal)
  elif trueVal < 0.4:
    h_pur_trueBin1.Fill(predVal)
  elif trueVal < 0.6:
    h_pur_trueBin2.Fill(predVal)
  elif trueVal < 0.8:
    h_pur_trueBin3.Fill(predVal)
  else:
    h_pur_trueBin4.Fill(predVal)



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
    testProcCorrect = 0
    testCorrect_p0 = 0
    testCorrect_p1 = 0
    testCorrect_p2 = 0
    total_p0 = 0
    total_p1 = 0
    total_p2 = 0
    testSteps = len(dataloader.dataset) // dataloader.batch_size
    tstep = 0
    
    effCounts = {0: classCounter(), 1: classCounter(), 2: classCounter(),
                 3: classCounter(), 4: classCounter(), 5: classCounter()} 
    purCounts = {0: classCounter(), 1: classCounter(), 2: classCounter(),
                 3: classCounter(), 4: classCounter(), 5: classCounter()} 

    effCountsComp = {0: fiveClassCounter(), 1: fiveClassCounter(), 2: fiveClassCounter(),
                     3: fiveClassCounter(), 4: fiveClassCounter()} 
    purCountsComp = {0: fiveClassCounter(), 1: fiveClassCounter(), 2: fiveClassCounter(),
                     3: fiveClassCounter(), 4: fiveClassCounter()} 

    effCountsPur = {0: fiveClassCounter(), 1: fiveClassCounter(), 2: fiveClassCounter(),
                     3: fiveClassCounter(), 4: fiveClassCounter()} 
    purCountsPur = {0: fiveClassCounter(), 1: fiveClassCounter(), 2: fiveClassCounter(),
                     3: fiveClassCounter(), 4: fiveClassCounter()} 

    effCountsProc = {0: threeClassCounter(), 1: threeClassCounter(), 2: threeClassCounter()}
    purCountsProc = {0: threeClassCounter(), 1: threeClassCounter(), 2: threeClassCounter()}

    with torch.no_grad():
        
        for batch, (X, y) in enumerate(dataloader):
            if tstep % 10 == 0:
                print("reached validation batch %i of %i"%(tstep, testSteps), flush=True)
            if args.multiTask or args.tripleTask or args.quadTask:
                X = X.to(args.device)
                outputs = model(X)
                pred = outputs[0]
                pred_comp = outputs[1]
                if args.tripleTask or args.quadTask:
                  pred_pur = outputs[2]
                  yPur = y[2]
                  yPurCl = torch.LongTensor([getCompClass(y[2][i].item()) for i in range(y[2].size(0))])
                  if args.batch_size == 1:
                    yPurCl_pred = torch.LongTensor([getCompClass(pred_pur.item())])
                  else:
                    yPurCl_pred = torch.LongTensor([getCompClass(pred_pur[i].item()) for i in range(pred_pur.size(0))])
                if args.classifyComp:
                  yCompCl = y[1].type(torch.LongTensor)
                  yCompCl_pred = pred_comp.argmax(1)
                else:
                  yComp = y[1]
                  yCompCl = torch.LongTensor([getCompClass(y[1][i].item()) for i in range(y[1].size(0))])
                  if args.batch_size == 1:
                    yCompCl_pred = torch.LongTensor([getCompClass(pred_comp.item())])
                  else:
                    yCompCl_pred = torch.LongTensor([getCompClass(pred_comp[i].item()) for i in range(pred_comp.size(0))])
                if args.quadTask:
                  yProc = y[3].type(torch.LongTensor).to(args.device)
                  pred_proc = outputs[3].to(args.device)
                  yProc_pred = pred_proc.argmax(1)
                y = y[0].type(torch.LongTensor)
                y, yCompCl, yCompCl_pred = y.to(args.device), yCompCl.to(args.device), yCompCl_pred.to(args.device)
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

            if args.quadTask:
                for i in range(yProc.size(0)): 
                  effCountsProc[yProc[i].item()].update(yProc_pred[i].item())
                  purCountsProc[yProc_pred[i].item()].update(yProc[i].item())

            if args.multiTask or args.tripleTask or args.quadTask:
                for i in range(yCompCl.size(0)):
                  effCountsComp[yCompCl[i].item()].update(yCompCl_pred[i].item())
                  purCountsComp[yCompCl_pred[i].item()].update(yCompCl[i].item())
                  if args.batch_size == 1:
                    fillCompHistos(yComp[i].item(), pred_comp.item(), yCompCl[i].item(), yCompCl_pred[i].item())
                  else:
                    fillCompHistos(yComp[i].item(), pred_comp[i].item(), yCompCl[i].item(), yCompCl_pred[i].item())
                if args.tripleTask or args.quadTask:
                    for i in range(yPurCl.size(0)):
                      effCountsPur[yPurCl[i].item()].update(yPurCl_pred[i].item())
                      purCountsPur[yPurCl_pred[i].item()].update(yPurCl[i].item())
                      if args.batch_size == 1:
                        fillPurityHistos(yPur[i].item(), pred_pur.item(), yPurCl[i].item(), yPurCl_pred[i].item())
                      else:
                        fillPurityHistos(yPur[i].item(), pred_pur[i].item(), yPurCl[i].item(), yPurCl_pred[i].item())
            
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

            if args.quadTask:

                iP0 = (yProc == 0).nonzero(as_tuple=True)
                iP1 = (yProc == 1).nonzero(as_tuple=True)
                iP2 = (yProc == 2).nonzero(as_tuple=True)

                total_p0 += yProc[iP0].size(dim=0)
                total_p1 += yProc[iP1].size(dim=0)
                total_p2 += yProc[iP2].size(dim=0)

                testProcCorrect += (pred_proc.argmax(1) == yProc).type(torch.float).sum().item()
                if yProc[iP0].size(dim=0) > 0:
                    testCorrect_p0 += (pred_proc[iP0].argmax(1) == yProc[iP0]).type(torch.float).sum().item()
                if yProc[iP1].size(dim=0) > 0:
                    testCorrect_p1 += (pred_proc[iP1].argmax(1) == yProc[iP1]).type(torch.float).sum().item()
                if yProc[iP2].size(dim=0) > 0:
                    testCorrect_p2 += (pred_proc[iP2].argmax(1) == yProc[iP2]).type(torch.float).sum().item()

            tstep += 1
            
    testAcc = testCorrect / len(dataloader.dataset)
    testAcc_e = testCorrect_e / total_e
    testAcc_ph = testCorrect_ph / total_ph
    testAcc_mu = testCorrect_mu / total_mu
    testAcc_pi = testCorrect_pi / total_pi
    testAcc_pr = testCorrect_pr / total_pr
    testAcc_o = testCorrect_o / total_o if (total_o > 0) else 0.
    testProcAcc = testProcCorrect / len(dataloader.dataset)
    testProcAcc_p0 = testCorrect_p0 / total_p0 if (total_p0 > 0) else -1.
    testProcAcc_p1 = testCorrect_p1 / total_p1 if (total_p1 > 0) else -1.
    testProcAcc_p2 = testCorrect_p2 / total_p2 if (total_p2 > 0) else -1.
    
    return testAcc, testAcc_e, testAcc_ph, testAcc_mu, testAcc_pi, testAcc_pr, testAcc_o, effCounts, purCounts, effCountsComp, purCountsComp, effCountsPur, purCountsPur, effCountsProc, purCountsProc, testProcAcc, testProcAcc_p0, testProcAcc_p1, testProcAcc_p2





teA, teA_e, teA_ph, teA_mu, teA_pi, teA_pr, teA_o, effCounts, purCounts, effCountsComp, purCountsComp, effCountsPur, purCountsPur, effCountsProc, purCountsProc, tePA, tePA_0, tePA_1, tePA_2  = test(dataloader, model)
print("test accuracy:", teA, " electron test accuracy:", teA_e, " photon test accuracy:", teA_ph, " muon test accuracy:", teA_mu, " pion test accuracy:", teA_pi, " proton test accuracy:", teA_pr, " other test accuracy:", teA_o, " process accuracy:", tePA, " primary process accuracy:", tePA_0, " neutral parent accuracy:", tePA_1, " charged parent accuracy:", tePA_2, flush=True)

print("PARTICLE CLASSIFICATION RESULTS:")
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

print()
print()

print("COMPLETENESS CLASSIFICATION RESULTS:")
for i in range(5):
    print("true class %i matches (%i instances):"%(i, effCountsComp[i].n))
    for j in range(5):
        ratio = 0.
        if effCountsComp[i].n > 0:
            ratio = effCountsComp[i].n_match[j]/effCountsComp[i].n
        print("    %i: %f"%(j, ratio))
for i in range(5):
    print("pred class %i matches (%i instances):"%(i, purCountsComp[i].n))
    for j in range(5):
        ratio = 0.
        if purCountsComp[i].n > 0:
            ratio = purCountsComp[i].n_match[j]/purCountsComp[i].n
        print("    %i: %f"%(j, ratio))

print()
print()

print("PURITY CLASSIFICATION RESULTS:")
for i in range(5):
    print("true class %i matches (%i instances):"%(i, effCountsPur[i].n))
    for j in range(5):
        ratio = 0.
        if effCountsPur[i].n > 0:
            ratio = effCountsPur[i].n_match[j]/effCountsPur[i].n
        print("    %i: %f"%(j, ratio))
for i in range(5):
    print("pred class %i matches (%i instances):"%(i, purCountsPur[i].n))
    for j in range(5):
        ratio = 0.
        if purCountsPur[i].n > 0:
            ratio = purCountsPur[i].n_match[j]/purCountsPur[i].n
        print("    %i: %f"%(j, ratio))

print()
print()

print("PROCESS CLASSIFICATION RESULTS:")
for i in range(3):
    print("true class %i matches (%i instances):"%(i, effCountsProc[i].n))
    for j in range(3):
        ratio = 0.
        if effCountsProc[i].n > 0:
            ratio = effCountsProc[i].n_match[j]/effCountsProc[i].n
        print("    %i: %f"%(j, ratio))
for i in range(3):
    print("pred class %i matches (%i instances):"%(i, purCountsProc[i].n))
    for j in range(3):
        ratio = 0.
        if purCountsProc[i].n > 0:
            ratio = purCountsProc[i].n_match[j]/purCountsProc[i].n
        print("    %i: %f"%(j, ratio))



outFile = rt.TFile(args.outfile, "RECREATE")

cnv_comp_heatmap = rt.TCanvas("cnv_comp_heatmap")
h_comp_heatmap.Draw("COLZ")
cnv_comp_heatmap.Write()

cnv_pur_heatmap = rt.TCanvas("cnv_pur_heatmap")
h_pur_heatmap.Draw("COLZ")
cnv_pur_heatmap.Write()

cnv_comp_predBins = rt.TCanvas("cnv_comp_predBins")
h_comp_predBin4.Draw("EHIST")
h_comp_predBin3.Draw("EHISTSAME")
h_comp_predBin2.Draw("EHISTSAME")
h_comp_predBin1.Draw("EHISTSAME")
h_comp_predBin0.Draw("EHISTSAME")
leg_comp_predBins = rt.TLegend(0.7,0.7,0.9,0.9)
leg_comp_predBins.AddEntry(h_comp_predBin0, "0.0 < predicted completeness < 0.2", "l")
leg_comp_predBins.AddEntry(h_comp_predBin1, "0.2 < predicted completeness < 0.4", "l")
leg_comp_predBins.AddEntry(h_comp_predBin2, "0.4 < predicted completeness < 0.6", "l")
leg_comp_predBins.AddEntry(h_comp_predBin3, "0.6 < predicted completeness < 0.8", "l")
leg_comp_predBins.AddEntry(h_comp_predBin4, "0.8 < predicted completeness < 1.0", "l")
leg_comp_predBins.Draw()
cnv_comp_predBins.Write()

cnv_comp_trueBins = rt.TCanvas("cnv_comp_trueBins")
h_comp_trueBin4.Draw("EHIST")
h_comp_trueBin3.Draw("EHISTSAME")
h_comp_trueBin2.Draw("EHISTSAME")
h_comp_trueBin1.Draw("EHISTSAME")
h_comp_trueBin0.Draw("EHISTSAME")
leg_comp_trueBins = rt.TLegend(0.7,0.7,0.9,0.9)
leg_comp_trueBins.AddEntry(h_comp_trueBin0, "0.0 < true completeness < 0.2", "l")
leg_comp_trueBins.AddEntry(h_comp_trueBin1, "0.2 < true completeness < 0.4", "l")
leg_comp_trueBins.AddEntry(h_comp_trueBin2, "0.4 < true completeness < 0.6", "l")
leg_comp_trueBins.AddEntry(h_comp_trueBin3, "0.6 < true completeness < 0.8", "l")
leg_comp_trueBins.AddEntry(h_comp_trueBin4, "0.8 < true completeness < 1.0", "l")
leg_comp_trueBins.Draw()
cnv_comp_trueBins.Write()

cnv_pur_predBins = rt.TCanvas("cnv_pur_predBins")
h_pur_predBin4.Draw("EHIST")
h_pur_predBin3.Draw("EHISTSAME")
h_pur_predBin2.Draw("EHISTSAME")
if not args.noLowPurBins:
  h_pur_predBin1.Draw("EHISTSAME")
  #h_pur_predBin0.Draw("EHISTSAME")
leg_pur_predBins = rt.TLegend(0.7,0.7,0.9,0.9)
if not args.noLowPurBins:
  #leg_pur_predBins.AddEntry(h_pur_predBin0, "0.0 < predicted purity < 0.2", "l")
  leg_pur_predBins.AddEntry(h_pur_predBin1, "0.2 < predicted purity < 0.4", "l")
leg_pur_predBins.AddEntry(h_pur_predBin2, "0.4 < predicted purity < 0.6", "l")
leg_pur_predBins.AddEntry(h_pur_predBin3, "0.6 < predicted purity < 0.8", "l")
leg_pur_predBins.AddEntry(h_pur_predBin4, "0.8 < predicted purity < 1.0", "l")
leg_pur_predBins.Draw()
cnv_pur_predBins.Write()

cnv_pur_trueBins = rt.TCanvas("cnv_pur_trueBins")
h_pur_trueBin4.Draw("EHIST")
h_pur_trueBin3.Draw("EHISTSAME")
if not args.noLowPurBins:
  h_pur_trueBin2.Draw("EHISTSAME")
  h_pur_trueBin1.Draw("EHISTSAME")
  #h_pur_trueBin0.Draw("EHISTSAME")
leg_pur_trueBins = rt.TLegend(0.7,0.7,0.9,0.9)
if not args.noLowPurBins:
  #leg_pur_trueBins.AddEntry(h_pur_trueBin0, "0.0 < true purity < 0.2", "l")
  leg_pur_trueBins.AddEntry(h_pur_trueBin1, "0.2 < true purity < 0.4", "l")
  leg_pur_trueBins.AddEntry(h_pur_trueBin2, "0.4 < true purity < 0.6", "l")
leg_pur_trueBins.AddEntry(h_pur_trueBin3, "0.6 < true purity < 0.8", "l")
leg_pur_trueBins.AddEntry(h_pur_trueBin4, "0.8 < true purity < 1.0", "l")
leg_pur_trueBins.Draw()
cnv_pur_trueBins.Write()


h_comp_predBin0_norm = h_comp_predBin0.Clone("h_comp_predBin0_norm")
h_comp_predBin1_norm = h_comp_predBin1.Clone("h_comp_predBin1_norm")
h_comp_predBin2_norm = h_comp_predBin2.Clone("h_comp_predBin2_norm")
h_comp_predBin3_norm = h_comp_predBin3.Clone("h_comp_predBin3_norm")
h_comp_predBin4_norm = h_comp_predBin4.Clone("h_comp_predBin4_norm")
h_comp_trueBin0_norm = h_comp_trueBin0.Clone("h_comp_trueBin0_norm")
h_comp_trueBin1_norm = h_comp_trueBin1.Clone("h_comp_trueBin1_norm")
h_comp_trueBin2_norm = h_comp_trueBin2.Clone("h_comp_trueBin2_norm")
h_comp_trueBin3_norm = h_comp_trueBin3.Clone("h_comp_trueBin3_norm")
h_comp_trueBin4_norm = h_comp_trueBin4.Clone("h_comp_trueBin4_norm")
if not args.noLowPurBins:
  h_pur_predBin0_norm = h_pur_predBin0.Clone("h_pur_predBin0_norm")
  h_pur_predBin1_norm = h_pur_predBin1.Clone("h_pur_predBin1_norm")
h_pur_predBin2_norm = h_pur_predBin2.Clone("h_pur_predBin2_norm")
h_pur_predBin3_norm = h_pur_predBin3.Clone("h_pur_predBin3_norm")
h_pur_predBin4_norm = h_pur_predBin4.Clone("h_pur_predBin4_norm")
if not args.noLowPurBins:
  h_pur_trueBin0_norm = h_pur_trueBin0.Clone("h_pur_trueBin0_norm")
  h_pur_trueBin1_norm = h_pur_trueBin1.Clone("h_pur_trueBin1_norm")
  h_pur_trueBin2_norm = h_pur_trueBin2.Clone("h_pur_trueBin2_norm")
h_pur_trueBin3_norm = h_pur_trueBin3.Clone("h_pur_trueBin3_norm")
h_pur_trueBin4_norm = h_pur_trueBin4.Clone("h_pur_trueBin4_norm")

h_comp_predBin0_norm.GetYaxis().SetTitle("area normalized event count")
h_comp_predBin1_norm.GetYaxis().SetTitle("area normalized event count")
h_comp_predBin2_norm.GetYaxis().SetTitle("area normalized event count")
h_comp_predBin3_norm.GetYaxis().SetTitle("area normalized event count")
h_comp_predBin4_norm.GetYaxis().SetTitle("area normalized event count")
h_comp_trueBin0_norm.GetYaxis().SetTitle("area normalized event count")
h_comp_trueBin1_norm.GetYaxis().SetTitle("area normalized event count")
h_comp_trueBin2_norm.GetYaxis().SetTitle("area normalized event count")
h_comp_trueBin3_norm.GetYaxis().SetTitle("area normalized event count")
h_comp_trueBin4_norm.GetYaxis().SetTitle("area normalized event count")
if not args.noLowPurBins:
  h_pur_predBin0_norm.GetYaxis().SetTitle("area normalized event count")
  h_pur_predBin1_norm.GetYaxis().SetTitle("area normalized event count")
h_pur_predBin2_norm.GetYaxis().SetTitle("area normalized event count")
h_pur_predBin3_norm.GetYaxis().SetTitle("area normalized event count")
h_pur_predBin4_norm.GetYaxis().SetTitle("area normalized event count")
if not args.noLowPurBins:
  h_pur_trueBin0_norm.GetYaxis().SetTitle("area normalized event count")
  h_pur_trueBin1_norm.GetYaxis().SetTitle("area normalized event count")
  h_pur_trueBin2_norm.GetYaxis().SetTitle("area normalized event count")
h_pur_trueBin3_norm.GetYaxis().SetTitle("area normalized event count")
h_pur_trueBin4_norm.GetYaxis().SetTitle("area normalized event count")

h_comp_predBin0_norm.Scale(1.0/h_comp_predBin0_norm.Integral())
h_comp_predBin1_norm.Scale(1.0/h_comp_predBin1_norm.Integral())
h_comp_predBin2_norm.Scale(1.0/h_comp_predBin2_norm.Integral())
h_comp_predBin3_norm.Scale(1.0/h_comp_predBin3_norm.Integral())
h_comp_predBin4_norm.Scale(1.0/h_comp_predBin4_norm.Integral())
h_comp_trueBin0_norm.Scale(1.0/h_comp_trueBin0_norm.Integral())
h_comp_trueBin1_norm.Scale(1.0/h_comp_trueBin1_norm.Integral())
h_comp_trueBin2_norm.Scale(1.0/h_comp_trueBin2_norm.Integral())
h_comp_trueBin3_norm.Scale(1.0/h_comp_trueBin3_norm.Integral())
h_comp_trueBin4_norm.Scale(1.0/h_comp_trueBin4_norm.Integral())
if not args.noLowPurBins:
  h_pur_predBin0_norm.Scale(1.0/h_pur_predBin0_norm.Integral())
  h_pur_predBin1_norm.Scale(1.0/h_pur_predBin1_norm.Integral())
h_pur_predBin2_norm.Scale(1.0/h_pur_predBin2_norm.Integral())
h_pur_predBin3_norm.Scale(1.0/h_pur_predBin3_norm.Integral())
h_pur_predBin4_norm.Scale(1.0/h_pur_predBin4_norm.Integral())
if not args.noLowPurBins:
  h_pur_trueBin0_norm.Scale(1.0/h_pur_trueBin0_norm.Integral())
  h_pur_trueBin1_norm.Scale(1.0/h_pur_trueBin1_norm.Integral())
  h_pur_trueBin2_norm.Scale(1.0/h_pur_trueBin2_norm.Integral())
h_pur_trueBin3_norm.Scale(1.0/h_pur_trueBin3_norm.Integral())
h_pur_trueBin4_norm.Scale(1.0/h_pur_trueBin4_norm.Integral())

cnv_comp_predBins_norm = rt.TCanvas("cnv_comp_predBins_norm")
h_comp_predBin4_norm.Draw("EHIST")
h_comp_predBin3_norm.Draw("EHISTSAME")
h_comp_predBin2_norm.Draw("EHISTSAME")
h_comp_predBin1_norm.Draw("EHISTSAME")
h_comp_predBin0_norm.Draw("EHISTSAME")
leg_comp_predBins_norm = rt.TLegend(0.7,0.7,0.9,0.9)
leg_comp_predBins_norm.AddEntry(h_comp_predBin0_norm, "0.0 < predicted completeness < 0.2", "l")
leg_comp_predBins_norm.AddEntry(h_comp_predBin1_norm, "0.2 < predicted completeness < 0.4", "l")
leg_comp_predBins_norm.AddEntry(h_comp_predBin2_norm, "0.4 < predicted completeness < 0.6", "l")
leg_comp_predBins_norm.AddEntry(h_comp_predBin3_norm, "0.6 < predicted completeness < 0.8", "l")
leg_comp_predBins_norm.AddEntry(h_comp_predBin4_norm, "0.8 < predicted completeness < 1.0", "l")
leg_comp_predBins_norm.Draw()
cnv_comp_predBins_norm.Write()

cnv_comp_trueBins_norm = rt.TCanvas("cnv_comp_trueBins_norm")
h_comp_trueBin4_norm.Draw("EHIST")
h_comp_trueBin3_norm.Draw("EHISTSAME")
h_comp_trueBin2_norm.Draw("EHISTSAME")
h_comp_trueBin1_norm.Draw("EHISTSAME")
h_comp_trueBin0_norm.Draw("EHISTSAME")
leg_comp_trueBins_norm = rt.TLegend(0.7,0.7,0.9,0.9)
leg_comp_trueBins_norm.AddEntry(h_comp_trueBin0_norm, "0.0 < true completeness < 0.2", "l")
leg_comp_trueBins_norm.AddEntry(h_comp_trueBin1_norm, "0.2 < true completeness < 0.4", "l")
leg_comp_trueBins_norm.AddEntry(h_comp_trueBin2_norm, "0.4 < true completeness < 0.6", "l")
leg_comp_trueBins_norm.AddEntry(h_comp_trueBin3_norm, "0.6 < true completeness < 0.8", "l")
leg_comp_trueBins_norm.AddEntry(h_comp_trueBin4_norm, "0.8 < true completeness < 1.0", "l")
leg_comp_trueBins_norm.Draw()
cnv_comp_trueBins_norm.Write()

cnv_pur_predBins_norm = rt.TCanvas("cnv_pur_predBins_norm")
h_pur_predBin4_norm.Draw("EHIST")
h_pur_predBin3_norm.Draw("EHISTSAME")
h_pur_predBin2_norm.Draw("EHISTSAME")
if not args.noLowPurBins:
  h_pur_predBin1_norm.Draw("EHISTSAME")
  #h_pur_predBin0_norm.Draw("EHISTSAME")
leg_pur_predBins_norm = rt.TLegend(0.7,0.7,0.9,0.9)
if not args.noLowPurBins:
  #leg_pur_predBins_norm.AddEntry(h_pur_predBin0_norm, "0.0 < predicted purity < 0.2", "l")
  leg_pur_predBins_norm.AddEntry(h_pur_predBin1_norm, "0.2 < predicted purity < 0.4", "l")
leg_pur_predBins_norm.AddEntry(h_pur_predBin2_norm, "0.4 < predicted purity < 0.6", "l")
leg_pur_predBins_norm.AddEntry(h_pur_predBin3_norm, "0.6 < predicted purity < 0.8", "l")
leg_pur_predBins_norm.AddEntry(h_pur_predBin4_norm, "0.8 < predicted purity < 1.0", "l")
leg_pur_predBins_norm.Draw()
cnv_pur_predBins_norm.Write()

cnv_pur_trueBins_norm = rt.TCanvas("cnv_pur_trueBins_norm")
h_pur_trueBin4_norm.Draw("EHIST")
h_pur_trueBin3_norm.Draw("EHISTSAME")
if not args.noLowPurBins:
  h_pur_trueBin2_norm.Draw("EHISTSAME")
  h_pur_trueBin1_norm.Draw("EHISTSAME")
  #h_pur_trueBin0_norm.Draw("EHISTSAME")
leg_pur_trueBins_norm = rt.TLegend(0.7,0.7,0.9,0.9)
if not args.noLowPurBins:
  #leg_pur_trueBins_norm.AddEntry(h_pur_trueBin0_norm, "0.0 < true purity < 0.2", "l")
  leg_pur_trueBins_norm.AddEntry(h_pur_trueBin1_norm, "0.2 < true purity < 0.4", "l")
  leg_pur_trueBins_norm.AddEntry(h_pur_trueBin2_norm, "0.4 < true purity < 0.6", "l")
leg_pur_trueBins_norm.AddEntry(h_pur_trueBin3_norm, "0.6 < true purity < 0.8", "l")
leg_pur_trueBins_norm.AddEntry(h_pur_trueBin4_norm, "0.8 < true purity < 1.0", "l")
leg_pur_trueBins_norm.Draw()
cnv_pur_trueBins_norm.Write()

