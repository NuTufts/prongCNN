
import argparse
import sys
import os

import numpy as np
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


parser = argparse.ArgumentParser("evaluate Prong CNN")
parser.add_argument("-i", "--images_file", type=str, required=True, help="validation images file")
parser.add_argument("-m", "--model_path", type=str, required=True, help="model name")
parser.add_argument("-d", "--device", type=str, default="cuda", help="gpu/cpu device")
parser.add_argument("-n", "--num_workers", type=int, default=12, help="number of cpu workers for data loading")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="data loader batch size")
parser.add_argument("-o", "--outfile", type=str, default="", help="output pdf file name")
parser.add_argument("--multiGPU", action="store_true", help="use multiple GPUs")
parser.add_argument("--interactive", action="store_true", help="display plots one by one rather than saving to pdf")
parser.add_argument("--writeScores", action="store_true", help="plot and write score histograms")
args = parser.parse_args()

sys.path.append(args.model_path[:args.model_path.find("/checkpoints")])
from models_instanceNorm_reco_2chan_tripleTask import ResBlock, ResNet34
from datasets_reco_5ClassHardLabel_tripleTask import ProngDataset, mean, std

mpl.rcParams['figure.dpi'] = 300
if not args.interactive:
  plotfilename = args.outfile
  if plotfilename == "":
    plotfilename = args.images_file.replace(".root","_images.pdf")
  outpdf = matplotlib.backends.backend_pdf.PdfPages(plotfilename)

transform = transforms.Normalize(mean, std)
dataset = ProngDataset(args.images_file, transformations=transform, clip=4.0)
dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=args.num_workers)

model = ResNet34(2, ResBlock, outputs=5)
if "cuda" in args.device and args.multiGPU:
  model = nn.DataParallel(model)
if args.device == "cpu":
  checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
else:
  checkpoint = torch.load(args.model_path)
try:
  model.load_state_dict(checkpoint['model_state_dict'])
except:
  model.module.load_state_dict(checkpoint['model_state_dict'])
model.to(args.device)
model.eval()


#def plotImage(X, r, sr, e, pdg, purity, completeness, visE, compPred, purPred, elScore, phScore, muScore, piScore, prScore):
def plotImage(X, compPred, purPred, elScore, phScore, muScore, piScore, prScore):
  #pltmin = None
  #pltmax = None
  pltmin = -1.
  pltmax = 2.
  suptitlesize=6
  titlesize=8
  ticksize=6
  X0 = X[0:2].reshape(2, 512, 512)
  X1 = X[2:4].reshape(2, 512, 512)
  X2 = X[4:].reshape(2, 512, 512)
  fig = plt.figure(0, clear=True)
  plt.subplot(2,3,1)
  plt.imshow(X0.numpy()[0], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 0 prong", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,2)
  plt.imshow(X1.numpy()[0], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 1 prong", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,3)
  plt.imshow(X2.numpy()[0], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 2 prong", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,4)
  plt.imshow(X0.numpy()[1], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 0 all", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,5)
  plt.imshow(X1.numpy()[1], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 1 all", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,6)
  plt.imshow(X2.numpy()[1], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 2 all", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  #plt.suptitle("Run %i Subrun %i Event %i  |  Prong: pdg %i, purity %.2f, completeness %.2f, visible energy %.2e \n e- score %.2f, photon score %.2f, mu score: %.2f, pi score %.2f, proton score %.2f, completeness prediction: %.2f, purity prediction: %.2f"%(r, sr, e, pdg, purity, completeness, visE, elScore, phScore, muScore, piScore, prScore, compPred, purPred), fontsize=suptitlesize)
  plt.suptitle("e- score %.2f, photon score %.2f, mu score: %.2f, pi score %.2f, proton score %.2f \n completeness prediction: %.2f, purity prediction: %.2f"%(elScore, phScore, muScore, piScore, prScore, compPred, purPred), fontsize=suptitlesize)
  if args.interactive:
    plt.show()
    input("Press Enter to continue...")
  else:
    outpdf.savefig(fig)
  return

def plotScores(scores):
  #suptitlesize=6
  titlesize=8
  ticksize=6
  fig = plt.figure(0, clear=True)
  plt.subplot(2,3,1)
  plt.hist(scores[0])
  plt.title("electron score distribution", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,2)
  plt.hist(scores[1])
  plt.title("photon score distribution", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,3)
  plt.hist(scores[2])
  plt.title("muon score distribution", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,4)
  plt.hist(scores[3])
  plt.title("pion score distribution", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,5)
  plt.hist(scores[4])
  plt.title("proton score distribution", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  if args.interactive:
    plt.show()
    input("Press Enter to continue...")
  else:
    outpdf.savefig(fig)
  return


if args.writeScores:
  scores = [ [], [], [], [], [] ]

for X, y in dataloader:
  X = X.to(args.device)
  outputs = model(X)
  for i in range(y[0].size(0)):
    elScore = outputs[0][i][0].item()
    phScore = outputs[0][i][1].item()
    muScore = outputs[0][i][2].item()
    piScore = outputs[0][i][3].item()
    prScore = outputs[0][i][4].item()
    if y[0].size(0) > 1:
      pred_comp = outputs[1][i].item()
      pred_pur = outputs[2][i].item()
    else:
      pred_comp = outputs[1].item()
      pred_pur = outputs[2].item()
    plotImage(X[i].cpu(), pred_comp, pred_pur, elScore, phScore, muScore, piScore, prScore)
    if args.writeScores:
      scores[0].append(elScore)
      scores[1].append(phScore)
      scores[2].append(muScore)
      scores[3].append(piScore)
      scores[4].append(prScore)

if args.writeScores:
  scores = np.array(scores)
  with open(args.images_file.replace(".root","_scoreArray.pkl"), 'wb') as f:
    pickle.dump(scores, f)

if not args.interactive:
  if args.writeScores:
    plotScores(scores)
  outpdf.close()


