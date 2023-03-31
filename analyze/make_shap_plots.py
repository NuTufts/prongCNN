
import sys, os
import argparse
import random

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/models')
from models_instanceNorm_reco_2chan_tripleTask_forSHAP import ResBlock, ResNet34
from datasets_reco_5ClassHardLabel_tripleTask import ProngDataset, mean, std

import shap

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


parser = argparse.ArgumentParser("make shap value plots")
parser.add_argument("-t", "--train_file", type=str, default="../train/images/prongCNN_reco_v2me05_images_file_run3bNuAndNueOverlays_preprocess_v02_cleaned_minHit10_noSecondaries_noPurityCut_2000PerClassVal_train.root", help="train images file")
parser.add_argument("-v", "--val_file", type=str, default="../train/images/prongCNN_reco_v2me05_images_file_run3bNuAndNueOverlays_preprocess_v02_cleaned_minHit10_noSecondaries_noPurityCut_2000PerClassVal_test.root", help="validation images file")
parser.add_argument("-m", "--model_path", type=str, default="../models/checkpoints/run3bOverlays_tripleTask_plAll_2inChan_5ClassHard_minHit10_b64_oneCycleLR_v2me05_noPCTrain/ResNet34_recoProng_5class_epoch20_withLossWeights.pt", help="model name")
parser.add_argument("-nB", "--nBackground", type=int, default=100, help="number of background images for shap calculations")
parser.add_argument("-sBS", "--shapBatchSize", type=int, default=10, help="number of images to use per shap calculation")
parser.add_argument("-nSB", "--nShapBatches", type=int, default=1, help="number of shap image batches to plot")
parser.add_argument("-mt", "--modelTask", type=int, default=0, help="plot shap values for this task (0: particle classification, 1: completeness regression, 2: purity regression")
parser.add_argument("-o", "--output", type=str, default="make_shap_plots_output.pdf", help="output pdf file path")
args = parser.parse_args()

device = torch.device('cpu')

outpdf_01 = matplotlib.backends.backend_pdf.PdfPages(args.output.replace(".pdf","_shapMax01.pdf"))
outpdf_05 = matplotlib.backends.backend_pdf.PdfPages(args.output.replace(".pdf","_shapMax05.pdf"))
outpdf_10 = matplotlib.backends.backend_pdf.PdfPages(args.output.replace(".pdf","_shapMax10.pdf"))
outpdf_20 = matplotlib.backends.backend_pdf.PdfPages(args.output.replace(".pdf","_shapMax20.pdf"))
outpdf_40 = matplotlib.backends.backend_pdf.PdfPages(args.output.replace(".pdf","_shapMax40.pdf"))


def getParticleName(partClass):
  if partClass == 0:
    return "electron"
  if partClass == 1:
    return "photon"
  if partClass == 2:
    return "muon"
  if partClass == 3:
    return "pion"
  if partClass == 4:
    return "proton"
  return "none"


imgMin = -1
imgMax = 2
suptitlesize=7
titlesize=7
ticksize=5

def printImage(image, shap, title, shapMin, shapMax, outpdf):
  fig = plt.figure(0, clear=True)
  plt.subplot(2,2,1)
  plt.imshow(image[0], vmin=imgMin, vmax=imgMax, cmap='jet')
  plt.title("prong image", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,2,2)
  im0 = plt.imshow(shap[0], vmin=shapMin, vmax=shapMax, cmap='seismic')
  cbar = plt.colorbar(im0)
  plt.title("prong shap values", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,2,3)
  plt.imshow(image[1], vmin=imgMin, vmax=imgMax, cmap='jet')
  plt.title("context image", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.subplot(2,2,4)
  im1 = plt.imshow(shap[1], vmin=shapMin, vmax=shapMax, cmap='seismic')
  cbar = plt.colorbar(im1)
  plt.title("context shap values", fontsize=titlesize)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)
  plt.suptitle(title, fontsize=suptitlesize)
  plt.subplots_adjust(hspace=0.25)
  outpdf.savefig(fig)
  

torch.manual_seed(1)
random.seed(0)
np.random.seed(0)

train_transform = transforms.Compose([transforms.Normalize(mean, std),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomVerticalFlip(0.5)])
test_transform = transforms.Normalize(mean, std)

image_dir = os.path.abspath('').replace("analyze","train/images")
train_dataset = ProngDataset(args.train_file, transformations=train_transform, clip=4.0)
test_dataset = ProngDataset(args.val_file, transformations=test_transform, clip=4.0)

shap_dataloader = DataLoader(train_dataset, batch_size=args.nBackground, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.shapBatchSize, shuffle=True)

model = ResNet34(2, ResBlock, outputs=5, returnTask=args.modelTask)

checkpoint = torch.load(args.model_path, map_location=device)
try:
  model.load_state_dict(checkpoint['model_state_dict'])
except:
  model.module.load_state_dict(checkpoint['model_state_dict'])
model.to("cpu")

background, _ = next(iter(shap_dataloader))
background = background.to(device)
explainer = shap.DeepExplainer(model, background)


for i in range(args.nShapBatches):

  print("calculating shap values for image %i"%i)

  images, targets = next(iter(test_dataloader))
  images = images.to(device)
  partClass, comp, purity = targets[0].to(device), targets[1].to(device), targets[2].to(device)
  shap_values = explainer.shap_values(images)

  for j in range(args.shapBatchSize):

    image0 = images[:,0:2].reshape(images.shape[0], 2, 512, 512).numpy()[j]
    image1 = images[:,2:4].reshape(images.shape[0], 2, 512, 512).numpy()[j]
    image2 = images[:,4:].reshape(images.shape[0], 2, 512, 512).numpy()[j]
    truthInfo = "(true %s prong with completeness: %.2f, purity: %.2f)"%(getParticleName(partClass[j]),comp[j],purity[j])

    for part in range(5):
      shap0 = shap_values[part][:,0:2].reshape(shap_values[part].shape[0], 2, 512, 512)[j]
      shap1 = shap_values[part][:,2:4].reshape(shap_values[part].shape[0], 2, 512, 512)[j]
      shap2 = shap_values[part][:,4:].reshape(shap_values[part].shape[0], 2, 512, 512)[j]
      printImage(image0, shap0, "plane 0 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.01, 0.01, outpdf_01)
      printImage(image1, shap1, "plane 1 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.01, 0.01, outpdf_01)
      printImage(image2, shap2, "plane 2 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.01, 0.01, outpdf_01)
      printImage(image0, shap0, "plane 0 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.05, 0.05, outpdf_05)
      printImage(image1, shap1, "plane 1 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.05, 0.05, outpdf_05)
      printImage(image2, shap2, "plane 2 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.05, 0.05, outpdf_05)
      printImage(image0, shap0, "plane 0 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.10, 0.10, outpdf_10)
      printImage(image1, shap1, "plane 1 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.10, 0.10, outpdf_10)
      printImage(image2, shap2, "plane 2 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.10, 0.10, outpdf_10)
      printImage(image0, shap0, "plane 0 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.20, 0.20, outpdf_20)
      printImage(image1, shap1, "plane 1 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.20, 0.20, outpdf_20)
      printImage(image2, shap2, "plane 2 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.20, 0.20, outpdf_20)
      printImage(image0, shap0, "plane 0 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.40, 0.40, outpdf_40)
      printImage(image1, shap1, "plane 1 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.40, 0.40, outpdf_40)
      printImage(image2, shap2, "plane 2 %s shap values\n%s"%(getParticleName(part),truthInfo), -0.40, 0.40, outpdf_40)

outpdf_01.close()
outpdf_05.close()
outpdf_10.close()
outpdf_20.close()
outpdf_40.close()

