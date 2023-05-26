# prongCNN

Files for training and evaluating a MicroBooNE prong CNN for the gen2 deep learning reconstruction framework

The latest iteration of this network performs particle classification, completeness regression, and purity regression for an input reconstructed prong.  
* completeness = the fraction of the true particle that is reconstructed in the input prong  
* purity = the fraction of the reconstructed prong that is actually from the true particle  
* In training/evaluation, "the true particle" = the simulated particle that deposited the most energy in the pixels belonging to the input prong (according to MC truth)

This repository has a variety of training scripts/options, dataset classes, and model classes that setup different network configurations that I've tested.

To get the current best-performing configurations, use:  
* The ProngDataset class (inherits from torch.utils.data.Dataset) from models/datasets_reco_5ClassHardLabel_tripleTask.py  
* The ResNet34 class (inherits from torch.nn.Module) from models/models_instanceNorm_reco_2chan_tripleTask.py

## Preprocessing

When training/evaluating the network, I use the following two scripts for preprocessing:  
preprocess/prepare_reco_images_cluster.py  
preprocess/split_image_file_by_val_num.py

### main preprocessing script

The first script (prepare_reco_images_cluster.py) is designed to be run on a cluster with slurm.  

You'll need to have the [ubdl repository](https://github.com/LArbys/ubdl) compiled and loaded in your environment  

It takes two inputs: a text file containing a list of DLgen2 kpsrecomanagerana larflow reco files, and a text file containing a matching list of merged_dlreco files.  

This script will need to be heavily modified to run in other reconstruction frameworks.  

The important thing is that the "ImageTree" root tree in the script's output file is structured the same way.  

Each entry in the ImageTree is for one prong (one input to the network).  

These are the ImageTree branches that are needed to actually train and run the network:  
* "pdg"
* "completeness"
* "purity"
* the input prong pixels (all branches beginning with "plane")
* the full-event/context pixels from the prong's cropped window (all branches beginning with "raw_plane")  

All of the other branches are for book keeping (e.g. run, subrun, event, and vertex/cluster ID numbers) or for studying the output prong sample.

### splitting output into training/validation samples
The second script (split_image_file_by_val_num.py) splits the output of prepare_reco_images_cluster.py into training and evaluation samples.  

It will put "args.nVal" prongs from each class into the evaluation sample and all remaining prongs in the training sample.

After preprocessing, there will be two root files: one containing all of the prong images and labels for prongs in the training sample, and one for the validation sample.

## Training

To train the network with the latest and greatest configuration, use:  
train/train_wandb_reco.py  
with the --tripleTask option

I was able to get the best results training for 20 epochs with a [one-cycle cosine annealing learning rate scheduler](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) with a min of 1e-8 and max of 1e-2  
To configure training with these options, use: -e 20 --schedOneCycleLR -l 1e-8 -lrM 1e-2  

I trained with a batch size of 64 using 12 cpus for data loading (options: -nbt 64 -nbv 64 -n 12)  

You'll also need to provide:
* The file paths for the training and validation samples produced during preprocessing with the --train_file and --val_file options
* The --model_path option to specify an output file path ending in ".pt" for the model checkpoints (these are saved at the end of every epoch, with ".pt" replaced with "_<epoch>.pt")
* The flag --singleGPU if training on only one GPU
* Optional but recommended: use the argument --runName to set the weights and biases run name (used for logging) </p>
  
## Evaluation

To evaluate the performance of the network using your full validation sample, you can use:  
analyze/evaluate_reco_model.py

This script will output information needed for particle classification confusion matrices, as well as some plots showing completeness/purity regression performance

You'll need to provide:
* The --tripleTask flag to load the correct model configuration
* The input sample with the --images_file option (this should be the validation sample file produced during the preprocessing step)
* The file path for the model checkpoint to load with the --model_path option (this is created during the training step)
* The file path for the output root file containing the completeness/purity performance plots
* The --singleGPU flag if running on a single GPU
* Optionally: --device to set the device for tensor calcuations (can use "cpu" to run on cpus)
* Optionally: --num_workers to set the number of cpus to use for data loading
* Optionally: --batch_size to set the batch size for data loading (affects speed but not performance) </p>

## Running the network without preprocessing

This section has information on running the network in an event loop without first running the preprocessing scripts

This requires:
* looping through events, grabbing the reconstructed track/shower you want to classify
* Calling a function to load the larcv images and select out pixels associated with the reco prong
* Load the prong images into a pytorch tensor
* Normalize the tensor (the pixel values) and pass it through the network </p>

Here is some example python code showing how to do this in the DLgen2 framework:  
(This code assumes you are running with the [ubdl repository](https://github.com/LArbys/ubdl) compiled and loaded in your environment)

```
import argparse

#import root:
import ROOT as rt

#if running in the DLgen2 framework with the ubdl repository, this module is needed to select and crop image pixels from input reco prongs:
from larflow import larflow

#load the larcv module from the ubdl repository to get the IO manager for reading in larcv images
from larcv import larcv

#import pytorch torch modules:
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

#Configure options 
parser = argparse.ArgumentParser("Script Title/Info")
#required:
parser.add_argument("-m", "--model_path", type=str, required=True, help="path to prong CNN checkpoint file")
#optional:
parser.add_argument("-d", "--device", type=str, default="cpu", help="gpu/cpu device")
parser.add_argument("--multiGPU", action="store_true", help="use multiple GPUs")
args = parser.parse_args()

#load the model class and mean, standard deviation for pixel normalization
#this code assumes the checkpoint file is stored in a "checkpoints" directory inside the models/ folder in this repo
sys.path.append(args.model_path[:args.model_path.find("/checkpoints")])
from models_instanceNorm_reco_2chan_tripleTask import ResBlock, ResNet34
from datasets_reco_5ClassHardLabel_tripleTask import mean, std

#load the model. the following code will get things to work regardless if running on cpu, single gpu, or multiple gpus
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

#load the DLGen2 class that does the pixel selection and cropping:
flowTriples = larflow.prep.FlowTriples()

#define the file with the reco information and the file with the larcv images

#DLgen2 larflowreco file:
kpsfile = rt.TFile("<file name>")
kpst = kpsfile.Get("KPSRecoManagerTree")

#matching larcv file with same entries:
iolcv = larcv.IOManager(larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward)
iolcv.add_in_file("<file name>")
iolcv.reverse_all_products()
iolcv.initialize()

#define function to convert CNN output class to PDG score
def getPID(cnnClass):
  if cnnClass == 0:
    return 11
  if cnnClass == 1:
    return 22
  if cnnClass == 2:
    return 13
  if cnnClass == 3:
    return 211
  if cnnClass == 4:
    return 2212
  return 0

#define function to create normalized pytorch tensor with prong images from vector of above-threshold pixels:
def makeImage(prong_vv):
  plane0pix_row = np.zeros(prong_vv[0].size(), dtype=int)
  plane0pix_col = np.zeros(prong_vv[0].size(), dtype=int)
  plane0pix_val = np.zeros(prong_vv[0].size(), dtype=float)
  plane1pix_row = np.zeros(prong_vv[1].size(), dtype=int)
  plane1pix_col = np.zeros(prong_vv[1].size(), dtype=int)
  plane1pix_val = np.zeros(prong_vv[1].size(), dtype=float)
  plane2pix_row = np.zeros(prong_vv[2].size(), dtype=int)
  plane2pix_col = np.zeros(prong_vv[2].size(), dtype=int)
  plane2pix_val = np.zeros(prong_vv[2].size(), dtype=float)
  raw_plane0pix_row = np.zeros(prong_vv[3].size(), dtype=int)
  raw_plane0pix_col = np.zeros(prong_vv[3].size(), dtype=int)
  raw_plane0pix_val = np.zeros(prong_vv[3].size(), dtype=float)
  raw_plane1pix_row = np.zeros(prong_vv[4].size(), dtype=int)
  raw_plane1pix_col = np.zeros(prong_vv[4].size(), dtype=int)
  raw_plane1pix_val = np.zeros(prong_vv[4].size(), dtype=float)
  raw_plane2pix_row = np.zeros(prong_vv[5].size(), dtype=int)
  raw_plane2pix_col = np.zeros(prong_vv[5].size(), dtype=int)
  raw_plane2pix_val = np.zeros(prong_vv[5].size(), dtype=float)
  for i, pix in enumerate(prong_vv[0]):
    plane0pix_row[i] = pix.row
    plane0pix_col[i] = pix.col
    plane0pix_val[i] = pix.val
  for i, pix in enumerate(prong_vv[1]):
    plane1pix_row[i] = pix.row
    plane1pix_col[i] = pix.col
    plane1pix_val[i] = pix.val
  for i, pix in enumerate(prong_vv[2]):
    plane2pix_row[i] = pix.row
    plane2pix_col[i] = pix.col
    plane2pix_val[i] = pix.val
  for i, pix in enumerate(prong_vv[3]):
    raw_plane0pix_row[i] = pix.row
    raw_plane0pix_col[i] = pix.col
    raw_plane0pix_val[i] = pix.val
  for i, pix in enumerate(prong_vv[4]):
    raw_plane1pix_row[i] = pix.row
    raw_plane1pix_col[i] = pix.col
    raw_plane1pix_val[i] = pix.val
  for i, pix in enumerate(prong_vv[5]):
    raw_plane2pix_row[i] = pix.row
    raw_plane2pix_col[i] = pix.col
    raw_plane2pix_val[i] = pix.val
  image = np.zeros((6,512,512))
  image[0, plane0pix_row, plane0pix_col] = plane0pix_val
  image[2, plane1pix_row, plane1pix_col] = plane1pix_val
  image[4, plane2pix_row, plane2pix_col] = plane2pix_val
  image[1, raw_plane0pix_row, raw_plane0pix_col] = raw_plane0pix_val
  image[3, raw_plane1pix_row, raw_plane1pix_col] = raw_plane1pix_val
  image[5, raw_plane2pix_row, raw_plane2pix_col] = raw_plane2pix_val
  image = torch.from_numpy(image).float()
  norm = transforms.Normalize(mean, std)
  image = norm(image).reshape(1,6,512,512)
  return torch.clamp(image, max=4.0)

#loop over entries:
for ientry in range(kpst.GetEntries()):

  iolcv.read_entry(ientry)
  kpst.GetEntry(ientry)
  
  #Double check run/subrun/event numbers to make sure two files are looking at same event!!
  
  #Get larcv image with all pixels:
  evtImage2D = iolcv.get_data(larcv.kProductImage2D, "wire")
  adc_v = evtImage2D.Image2DArray()
  #Get larcv image with wire-cell tagged cosmic pixels. These will be subtracted from images before running the network
  csmImage2D = iolcv.get_data(larcv.kProductImage2D, "thrumu")
  thrumu_v = csmImage2D.Image2DArray()
  
  #loop over vertices (kpst.nuvetoed_v in DLgen2) and select desired vertex
  vertex = <vurrent vertex in loop, identified neutrino vertex, or whatever vertex you want>
  
  #loop over reconstructed 3D clusters (tracks or showers):
  #In DLGen2, cluster_vector should be vertex.track_hitcluster_v when looping over tracks and vertex.shower_v if looping over showers
  for iC, cluster in enumerate(<cluster_vector>):
    #Image will be cropped to 512x512
    #We will try to fit entire cluster in image crop. If it won't fit, center image around specified crop point
    #Tracks and showers are handled differently for image cropping
    #If track: set crop point to end of track
    cropPt = vertex.track_v[iC].End()
    #If shower: set crop point to beginning of shower trunk:
    cropPt = vertex.shower_trunk_v[iC].Vertex()
    #Get above-threshold pixels to include in prong images:
    prong_vv = flowTriples.make_cropped_initial_sparse_prong_image_reco(adc_v,thrumu_v,trackCls,cropPt,10.,512,512)
    #Get tensor with prong images:
    prongImage = makeImage(prong_vv).to(args.device)
    #run network:
    prongCNN_out = model(prongImage)
    #Get prong CNN outputs:
    #predicted prong PDG:
    pid = getPID(prongCNN_out[0].argmax(1).item())
    #completeness prediction:
    comp = prongCNN_out[1].item()
    #purity prediction:
    purity = prongCNN_out[2].item()
    #individual particle scores (pid variable above assigned to particle type with highest score)
    electron_score = prongCNN_out[0][0][0].item()
    photon_score = prongCNN_out[0][0][1].item()
    muon_score = prongCNN_out[0][0][2].item()
    pion_score = prongCNN_out[0][0][3].item()
    proton_score = prongCNN_out[0][0][4].item()

```
