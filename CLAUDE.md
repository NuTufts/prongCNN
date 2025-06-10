# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

prongCNN is a MicroBooNE prong convolutional neural network for the gen2 deep learning reconstruction framework. It performs:
- Particle classification (electrons, photons, muons, pions, protons)
- Production process classification (primary, secondary with charged/neutral parent)
- Completeness regression (fraction of true particle reconstructed)
- Purity regression (fraction of reconstructed prong from true particle)

## Key Architecture

### Model Configuration
- Best model: `ResNet34` from `models/models_instanceNorm_reco_2chan_quadTask.py`
- Best dataset: `ProngDataset` from `models/datasets_reco_5ClassHardLabel_quadTask.py`
- Input: 6-channel 512x512 images (3 wire planes × 2 image types: prong + context)
- Normalization: Uses mean/std values defined in dataset classes

### Dependencies
- PyTorch for model training/inference
- ROOT for data handling
- ubdl repository (required for full functionality)
- larcv and larlite (for C++ interface)
- Weights & Biases for experiment tracking

## Common Development Commands

### Environment Setup
```bash
source setenv.sh
```

### Training
```bash
# Best configuration (20 epochs, one-cycle cosine annealing LR)
python train/train_wandb_reco.py \
    --train_file <training_file.root> \
    --val_file <validation_file.root> \
    --model_path <output_model.pt> \
    --quadTask \
    -e 20 \
    --schedOneCycleLR \
    -l 1e-8 \
    -lrM 1e-2 \
    -nbt 64 \
    -nbv 64 \
    -n 12 \
    --singleGPU \
    --runName <experiment_name>
```

### Evaluation
```bash
python analyze/evaluate_reco_model.py \
    --quadTask \
    --images_file <validation_file.root> \
    --model_path <checkpoint.pt> \
    --outfile <output_plots.root> \
    --singleGPU \
    --device cuda \
    --num_workers 12 \
    --batch_size 64
```

### Preprocessing Pipeline
1. **Generate prong images** (requires SLURM cluster):
   ```bash
   python preprocess/prepare_reco_images_cluster.py <reco_files.txt> <dlreco_files.txt>
   ```

2. **Apply pre-selection cuts**:
   ```bash
   python preprocess/clean_reco_image_file_5class.py \
       -i <input.root> \
       -o <output.root> \
       -pS 0.8 \    # min simulated particle fraction
       -nH 10 \     # min hits per plane
       -pD 0.0      # min dominant particle purity
   ```

3. **Split train/validation**:
   ```bash
   python preprocess/split_image_file_by_val_num.py \
       -i <input.root> \
       -o <output_prefix> \
       --nVal 2000  # per class for validation
   ```

### C++ Interface Build
```bash
cd larpid
mkdir -p build && cd build
cmake ..
make -j4
make install
```

## Code Organization

### Training Scripts (`train/`)
- Main training: `train_wandb_reco.py` with `--quadTask` flag
- Data loading: `dataloaders.py`
- Learning rate schedulers supported: StepLR, CyclicLR, CosineAnnealingWarmRestarts, OneCycleLR

### Model Definitions (`models/`)
- Model architectures: `models_instanceNorm_reco_2chan_*.py`
- Dataset classes: `datasets_reco_5ClassHardLabel_*.py`
- Various experimental configurations available

### Analysis Scripts (`analyze/`)
- Model evaluation: `evaluate_reco_model.py`
- Performance plotting: Various comparison scripts
- SHAP interpretability: `make_shap_plots.py`

### Preprocessing (`preprocess/`)
- Cluster job submission: `prepare_reco_images_cluster.py`
- Data cleaning: `clean_reco_image_file_5class.py`
- Train/val splitting: `split_image_file_by_val_num.py`

## Important Notes

1. **DLGen2 Framework**: Designed specifically for DLGen2 reconstruction framework
2. **ubdl Repository**: Required for pixel selection and image cropping functionality
3. **Checkpoint Format**: Models saved as PyTorch state dicts with epoch suffix
4. **Multi-GPU Support**: Use `--singleGPU` flag for single GPU, omit for multi-GPU
5. **Batch Processing**: Typical batch size is 64 for both training and validation
6. **Image Normalization**: Always apply normalization using dataset-specific mean/std values

## Pre-trained Model Location
On uboonegpvms: `/uboone/data/users/mmr/prongCNN/ResNet34_recoProng_5class_epoch20.pt`