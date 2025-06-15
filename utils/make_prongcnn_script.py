
import os,sys,argparse,time

"""
Script to make analysis ntuples for the uboone DL-gen2 reconstruction.

See README.md in the repo https://github.com/NuTufts/gen2ntuple
for definition of the different variables.

It also runs the LArPID CNN to classify prongs.

When provided metadata from the simulation, it also provides "truth" information about the neutrino interaction
and it's particles.

"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

sys.path.append("../models")

model_path = sys.argv[1]

from models_instanceNorm_reco_2chan_quadTask import ResBlock, ResNet34
from normalization_constants import mean, std

device = "cpu"
model = ResNet34(2, ResBlock, outputs=5)
#if "cuda" in args.device and args.multiGPU:
#  model = nn.DataParallel(model)
#if args.device == "cpu":
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

try:
  model.load_state_dict(checkpoint['model_state_dict'])
except:
  model.module.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(model)

scripted_model = torch.jit.script(model)

scripted_model.save("scripted_model.pt")

#with torch.no_grad():
  #print("make prong image: ",prong_vv.size(),flush=True)
  #prongImage = makeImage(prong_vv).to(args.device)
  #print("run prongCNN on track image",flush=True)
  #prongCNN_out = model(prongImage)

