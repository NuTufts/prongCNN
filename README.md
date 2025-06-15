# prongCNN

Files for training and evaluating a MicroBooNE prong CNN for the gen2 deep learning reconstruction framework

The latest iteration of this network performs particle classification, production process classification, completeness regression, and purity regression for an input reconstructed prong. 
* completeness = the fraction of the true particle that is reconstructed in the input prong  
* purity = the fraction of the reconstructed prong that is actually from the true particle  
* In training/evaluation, "the true particle" = the simulated particle that deposited the most energy in the pixels belonging to the input prong (according to MC truth)
* The defined particle classes are: electrons, photons, muons, pions, and protons
* The defined production process classes are: primary particle from neutrino interaction, secondary particle with a charged parent, and secondary particle with a neutral parent

This repository has a variety of training scripts/options, dataset classes, and model classes that setup different network configurations that I've tested.

To get the current best-performing configurations, use:  
* The ProngDataset class (inherits from torch.utils.data.Dataset) from models/datasets_reco_5ClassHardLabel_quadTask.py  
* The ResNet34 class (inherits from torch.nn.Module) from models/models_instanceNorm_reco_2chan_quadTask.py

TODO list:
* Provide interface function that receives IOManager and makes sparse image where one can choose to not veto shower-like prongs

## Preprocessing

When training/evaluating the network, I use the following three scripts for preprocessing:  
preprocess/prepare_reco_images_cluster.py  
preprocess/clean_reco_image_file_5class.py  
preprocess/split_image_file_by_val_num.py

### main preprocessing script

The first script (prepare_reco_images_cluster.py) is designed to be run on a cluster with slurm.  

You'll need to have the [ubdl repository](https://github.com/LArbys/ubdl) compiled and loaded in your environment  

The script takes two inputs: a text file containing a list of DLgen2 kpsrecomanagerana larflow reco files, and a text file containing a matching list of merged_dlreco files.  

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

### pre-selection script

The second script (clean_reco_image_file_5class.py) takes the output of prepare_reco_images_cluster.py and applies some basic pre-selection cuts to remove junk prongs:
* A minimum threshold for the fraction of the prong's visible energy that was produced by a simulated particle (removes cosmic prongs). Set this threshold with the -pS option.
* A minimum hit threshold. Prongs will be removed if any wire plane does not have at least this number of prong pixels. Set this threshold with the -nH option.
* A minimum dominant particle purity threshold. Prongs will be removed if there is not a single simulated particle that generates at least this fraction of the prong's visible energy. This is useful for ensuring one can apply a sensible true-particle-type label to each prong in the sample. Set this threshold with the -pD option.

### splitting output into training/validation samples

The third script (split_image_file_by_val_num.py) splits the output of clean_reco_image_file_5class.py into training and evaluation samples.  

It will put N prongs from each class into the evaluation sample and all remaining prongs in the training sample, where N is set by the --nVal option.

After preprocessing, there will be two root files: one containing all of the prong images and labels for prongs in the training sample, and one for the validation sample.

## Training

To train the network with the latest and greatest configuration, use:  
train/train_wandb_reco.py  
with the --quadTask option

I was able to get the best results training for 20 epochs with a [one-cycle cosine annealing learning rate scheduler](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) with a min of 1e-8 and max of 1e-2  
To configure training with these options, use: -e 20 --schedOneCycleLR -l 1e-8 -lrM 1e-2  

I trained with a batch size of 64 using 12 cpus for data loading (options: -nbt 64 -nbv 64 -n 12)  

You'll also need to provide:
* The file paths for the training and validation samples produced during preprocessing with the --train_file and --val_file options
* The --model_path option to specify an output file path ending in ".pt" for the model checkpoints (these are saved at the end of every epoch, with ".pt" replaced with "_\<epoch\>.pt")
* The flag --singleGPU if training on only one GPU
* Optional but recommended: use the argument --runName to set the weights and biases run name (used for logging) </p>
  
## Evaluation

To evaluate the performance of the network using your full validation sample, you can use:  
analyze/evaluate_reco_model.py

This script will output information needed for particle/process classification confusion matrices, as well as some plots showing completeness/purity regression performance

You'll need to provide:
* The --quadTask flag to load the correct model configuration
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
from models_instanceNorm_reco_2chan_quadTask import ResBlock, ResNet34
from datasets_reco_5ClassHardLabel_quadTask import mean, std

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

#define function to convert CNN output class to PDG
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
  vertex = <current vertex in loop, identified neutrino vertex, or whatever vertex you want>
  
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
    #predicted production process (0: primary, 1: secondary with neutral parent, 2: secondary with charged parent)
    process = prongCNN_out[3].argmax(1).item()
    #individual particle scores (pid variable above assigned to particle type with highest score)
    electron_score = prongCNN_out[0][0][0].item()
    photon_score = prongCNN_out[0][0][1].item()
    muon_score = prongCNN_out[0][0][2].item()
    pion_score = prongCNN_out[0][0][3].item()
    proton_score = prongCNN_out[0][0][4].item()
    #individual process scores (process variable above assigned to process type with highest score)
    primary_score = prongCNN_out[3][0][0].item()
    neutralParent_score = prongCNN_out[3][0][1].item()
    chargedParent_score = prongCNN_out[3][0][2].item()

```

## Running the network in different reconstruction frameworks (other than DLGen2)

The two main things to consider if attempting to run the network on prongs reconstructed in other reco frameworks are:
* Reading in LArCV images
* Coding up a method to select pixels associated with the prong

### Reading in LArCV images

LArCV images are stored in the merged_dlreco files generated for the DL gen1 reconstruction used in the eLEE analysis.

Check out the [LArCV repository](https://github.com/larbys/larcv/tree/ubdl_dev) for the code we use to read in LArCV images from merged_dlreco files. The readme has installation instructions.

There is a LArCV io manager defined in this repo that makes it easy to pull out the image objects.

For an example on how to use the io manager, take a look at the tutorial script: Tutorials/make_event_display.cxx from the [ubdl repository](https://github.com/LArbys/ubdl).
In particular, take a look at how the ev_in_adc_dlreco and img_2d_in_v objects are defined. The img_2d_in_v object is a vector of the three larcv images (for the three wire planes).
To get the vector of the three larcv images for wire-cell tagged (out-of-time) cosmic pixels only, replace the string "wire" in the io_larcv->get_data call used
to define ev_in_adc_dlreco with "thrumu". The following section discusses how to use the full event and cosmic larcv images to do the prong pixel selection in more detail.

Note that actually running this tutorial script will require installing the full ubdl repository (instructions are available in the repo's readme).

### Prong pixel selection

Once you have the reconstructed event from your reco framework loaded, have pulled out a reco prong to pass through the network, and have loaded in the "wire" and "thrumu" larcv images (discussed in the previous section) for the same event, you'll need to indentify the CNN image pixels for that prong.

To do this, I would suggest modifying the make_cropped_initial_sparse_prong_image_reco function from the larflow/larflow/PrepFlowMatchData/FlowTriples.(h,cxx) scripts in the [ubdl repository](https://github.com/LArbys/ubdl). This function takes the following input:
* the "wire" larcv image
* the "thrumu" larcv image
* the reconstructed prong object
* a point defining the center of the image crop
* a minimum pixel value threshold to throw out noise
* the row span for the cropped image
* the column span for the cropped image

To obtain the existing weights for the trained network, I used a minimm pixel value of 10 and 512 for the row and column spans (to get 512x512 images). You'll need to use those values when running with the pre-trained network.

If the input prong doesn't fit in a 512x512 image, the image is just centered on the input image crop point. I used the track end point for tracks and the shower start point for showers. This choice should be maintained if using the pre-trained network. If the prong does fit in the 512x512 image, the image is cropped (separately in each wire plane) around the middle of the prong.

After defining the pixel boundaries for the cropped image, this function fills and returns six vectors of CropPixData_t objects (see definition in FlowTriples.h).
Each of these CropPixData_t objects represents one pixel. There are six vectors from 3 wire planes x 2 image types (prong and full event).
The full event image vector for a given wire plane is filled for all pixels in the 512x512 image that are above the input threshold in the "wire" image
and below the input threshold in the "thrumu" image. This image is included to provide the network with context information. The prong image is filled with
pixels that satisfy the same threshold cuts and are also present in the input reco prong (the prong is reconstructed with a hit on the same wire and time tick).

Once you've modified this function to take as input a prong from a different reconstruction framework, you can use the output to:
* Define an image tensor to pass through the network, as shown in the makeImage function from the example code in the "Running the network without preprocessing" section
* Modify the preprocess/prepare_reco_images_cluster.py discussed in the "Preprocessing" section to generate a large prong image file to be used for either evaluating or re-training the network (as described in the "Training" and "Evaluation" sections).

## The Trained Network

The file containing the learned weights for the trained network can be found on the uboonegpvms at /uboone/data/users/mmr/prongCNN/ResNet34_recoProng_5class_epoch20.pt

This is what you can use as input for the --model_path arguments in the scripts from the "Evaluation" and "Running the network without preprocessing" sections above.

The network was trained using the options discussed in the "Training" section on all DLGen2 prongs from the run3b bnb nu and intrinsic nue overlay samples. More specifically, these input prongs were taken from the following samweb definitions:
* prodgenie_bnb_overlay_run3b_ssnet_wc_v2_nocrtremerge_run3b_ssnet_merged_dlreco
* prodgenie_bnb_intrinsic_nue_overlay_run3b_ssnet_wc_v2_nocrtremerge_run3b_ssnet_merged_dlreco

I ran all the reco prongs attached to reco neutrino vertices in these samples through the preprocessing scripts described in the "Preprocessing" section.

To evaluate the network, I pulled out 2000 prongs per particle class (for a total of 10,000 prongs) for the validation sample using the scripts described in that section.
Since I first aggregated the prong images in a single file before pulling out these 10,000 prongs for validation, I do not have separate
file lists for training and evaluation prongs. However, the prong image root files generated during preprocessing contain the run/subrun/event information
for every prong, so if needed you can take a look at the training and validation image files to see which events ended up in which sample (training or validation).

These training and validation prong image files can be found on the uboonegpvms at
* /uboone/data/users/mmr/prongCNN/prongCNN_reco_v2me05_images_file_run3bNuAndNueOverlays_preprocess_v03_cleaned_minHit10_noSecondaries_noPurityCut_2000PerClassVal_train.root
* /uboone/data/users/mmr/prongCNN/prongCNN_reco_v2me05_images_file_run3bNuAndNueOverlays_preprocess_v03_cleaned_minHit10_noSecondaries_noPurityCut_2000PerClassVal_test.root
* /uboone/data/users/mmr/prongCNN/prongCNN_reco_v2me05_images_file_run3bNuAndNueOverlays_preprocess_v03_cleaned_minHit10_noSecondaries_noPurityCut_2000PerClassVal_test_lowPurityRemoved.root

There is one training file and two validation files.
The difference comes from the cleaning script (clean_reco_image_file_5class.py) discussed in the "Preprocessing" section.
All three were produced with the options -pS 0.8 and -nH 10.
The training file and both validation files were all originally produced without a minimum dominant particle purity threshold (-pD 0).
The validation file with the "lowPurityRemoved" flag was passed back through the cleaning script with -pD 0.6 to remove prongs with purity < 60%.
I found that including the low purity prongs in the training sample did not worsen the network's performance on high-purity prongs, so they were included for training.
But you can't assign a robust particle label to a prong if it doesn't have a single simulated particle that generated the majority of it's visible energy, so I also
included a validation sample with a minimum dominant particle purity threshold of 60% to evaluate the network's particle classification performance (hence the two validation files).

