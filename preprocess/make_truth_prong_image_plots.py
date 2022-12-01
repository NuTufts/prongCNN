import ROOT as rt
import uproot

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
from torch.utils.data import Dataset, DataLoader, Sampler, Dataset
from torch.optim import AdamW
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import gc

outpdf = matplotlib.backends.backend_pdf.PdfPages("classification_challenge_truth_prong_images.pdf")
outtxt = open("classification_challenge_truth_prong_classes.txt","w")
outtxt.write("prong image number: truth matched class\n")

torch.manual_seed(1)
random.seed(0)
np.random.seed(0)

mean = (57.8182, 57.8182, 58.1807, 58.1807, 50.5312, 50.5312)
std = (62.9932, 62.9932, 62.6569, 62.6569, 42.0027, 42.0027)

meanPl2 = (50.5312, 50.5312)
stdPl2 = (42.0027, 42.0027)


def getClassName(cnnClass):
  if cnnClass == 0:
    return "electron"
  if cnnClass == 1:
    return "photon"
  if cnnClass == 2:
    return "muon"
  if cnnClass == 3:
    return "pion"
  if cnnClass == 4:
    return "proton"
  return "invalid class number"

def getClass(pid):
  if pid == 11: 
    return 0 
  if pid == 22: 
    return 1 
  if pid == 13: 
    return 2 
  if pid == 211:
    return 3
  if pid == 2212:
    return 4
  return 5


class ProngDataset(Dataset):
    
    def __init__(self, rootfile, transformations=None, clip=1000.0):
        self.file = uproot.open(rootfile)
        self.tree = self.file["ImageTree"]
        pdgs = self.tree["pdg"].array(library="np")
        self.classes = np.array([getClass(pdgs[i]) for i in range(len(pdgs))])
        self.transforms = transformations
        self.clipVal = clip
    
    def __getitem__(self, item):
        #print("retrieving ProngDataset entry", item)
        image = np.zeros((6,512,512))
    
        arrays = self.tree.arrays(["pdg", "plane0pix_row", "plane0pix_col", "plane0pix_val",
                                   "plane1pix_row", "plane1pix_col", "plane1pix_val",
                                   "plane2pix_row", "plane2pix_col", "plane2pix_val",
                                   "raw_plane0pix_row", "raw_plane0pix_col", "raw_plane0pix_val",
                                   "raw_plane1pix_row", "raw_plane1pix_col", "raw_plane1pix_val",
                                   "raw_plane2pix_row", "raw_plane2pix_col", "raw_plane2pix_val"],
                                  library="np", entry_start=item, entry_stop=item+1)
        image[0, arrays["plane0pix_row"][0], arrays["plane0pix_col"][0]] = arrays["plane0pix_val"][0]
        image[2, arrays["plane1pix_row"][0], arrays["plane1pix_col"][0]] = arrays["plane1pix_val"][0]
        image[4, arrays["plane2pix_row"][0], arrays["plane2pix_col"][0]] = arrays["plane2pix_val"][0]
        image[1, arrays["raw_plane0pix_row"][0], arrays["raw_plane0pix_col"][0]] = arrays["raw_plane0pix_val"][0]
        image[3, arrays["raw_plane1pix_row"][0], arrays["raw_plane1pix_col"][0]] = arrays["raw_plane1pix_val"][0]
        image[5, arrays["raw_plane2pix_row"][0], arrays["raw_plane2pix_col"][0]] = arrays["raw_plane2pix_val"][0]
    
        image = torch.from_numpy(image).float()
        Class = getClass(arrays["pdg"][0])
    
        if self.transforms is not None:
            image = self.transforms(image)
    
        return torch.clamp(image, max=self.clipVal), Class
    
    def __len__(self):
        return self.tree.num_entries


transform = transforms.Normalize(mean, std)
dataset = ProngDataset("prongCNN_images_file_5particle_wMasks_localData.root", transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6)

pltmin = -1.
pltmax = 2.
suptitlesize=16
titlesize=10
ticksize=6

classCounters = [0,0,0,0,0,0]

i = 0
for X, y in dataloader:
  #print(y[0].item(), X.shape, y.shape)
  if classCounters[y[0].item()] >= 15:
    continue
  if (classCounters[0] >= 15 and classCounters[1] >= 15 and classCounters[2] >= 15 and
      classCounters[3] >= 15 and classCounters[4] >= 15 and classCounters[5] >= 15):
    break
  classCounters[y[0].item()] += 1
  X0 = X[:,0:2].reshape(X.shape[0], 2, 512, 512)
  X1 = X[:,2:4].reshape(X.shape[0], 2, 512, 512)
  X2 = X[:,4:].reshape(X.shape[0], 2, 512, 512)
  fig = plt.figure(0, clear=True)
  plt.subplot(2,3,1)
  plt.imshow(X0.numpy()[0][0], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 0 prong", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,2)
  plt.imshow(X1.numpy()[0][0], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 1 prong", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,3)
  plt.imshow(X2.numpy()[0][0], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 2 prong", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,4)
  plt.imshow(X0.numpy()[0][1], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 0 all", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,5)
  plt.imshow(X1.numpy()[0][1], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 1 all", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,3,6)
  plt.imshow(X2.numpy()[0][1], vmin=pltmin, vmax=pltmax, cmap='jet')
  plt.title("plane 2 all", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.suptitle("prong image %i"%i, fontsize=suptitlesize)
  #plt.show()
  outpdf.savefig(fig)
  outtxt.write("%i: %s\n"%(i,getClassName(y[0].item())))
  i += 1

outpdf.close()
outtxt.close()


