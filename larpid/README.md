# LArPID - C++ Interface for prongCNN

LArPID provides a C++ interface to the prongCNN PyTorch model for particle identification in MicroBooNE's DLGen2 reconstruction framework. It enables efficient inference on prong-level reconstructed particles using LibTorch.

## Overview

LArPID is designed to:
- Load and run the prongCNN PyTorch model in C++
- Process reconstructed particle prongs from larflow clusters
- Generate cropped, sparse image representations for CNN input
- Perform particle classification and regression tasks
- Integrate seamlessly with the DLGen2 reconstruction chain

## Model Capabilities

The interface provides access to prongCNN's four tasks:
1. **Particle Classification**: Identifies particles as electrons, photons, muons, pions, or protons
2. **Process Classification**: Determines if particle is primary or secondary
3. **Completeness Regression**: Estimates fraction of true particle reconstructed
4. **Purity Regression**: Estimates fraction of reconstructed prong from true particle

## Dependencies

- **ROOT**: For data I/O and analysis
- **PyTorch C++ (LibTorch)**: For model inference
- **larcv**: For image data handling
- **larlite**: For reconstruction data structures
- **ublarcvapp**: For image processing utilities
- **larflow**: For prong/cluster data structures

Note: the version of LibTorch used must be built using the CXX-11 ABI standard. 
(e.g. It must have been built with _GLIBCXX_USE_CXX11_ABI_=1. 
Or ROOT and ubdl must be built with _GLIBCXX_USE_CXX11_ABI_=0).
Copies of libtorch that come with pip typically are built with _GLIBCXX_USE_CXX11_ABI_=0.
The easiest solution is to keep a separate libtorch copy that this builds with.
(Example of [website](https://download.pytorch.org/libtorch/cu111) download pre-compiled versions of pytorch with cxx-11 ABI.)

## Building

```bash
cd larpid
mkdir build && cd build
cmake ..
make -j4
make install
```

The build system will automatically:
- Detect ROOT and match its C++ standard
- Find PyTorch libraries and set appropriate RPATH
- Link against larcv, larlite, ublarcvapp, and larflow

## Directory Structure

```
larpid/
├── CMakeLists.txt          # Main build configuration
├── cmake/                  # CMake configuration files
├── larpid/
│   ├── app/               # Executable applications
│   │   ├── run_larpid.cxx       # Main inference executable
│   │   └── test_load_model.cxx  # Model loading test
│   ├── data/              # Data structures
│   │   ├── CropPixData_t.*      # Sparse pixel data format
│   │   └── ModelOutput.*        # CNN output container
│   ├── dataprep/          # Data preparation utilities
│   │   └── prepare_reco_images.* # Image preprocessing
│   ├── interface/         # Core interface code
│   │   └── LArPIDInterface.*    # Main interface functions
│   └── model/             # Model loading and inference
│       └── TorchModel.*         # PyTorch model wrapper
```

## Usage

### Basic Example

```cpp
#include "larpid/model/TorchModel.h"
#include "larpid/interface/LArPIDInterface.h"

// Load the model
larpid::model::TorchModel model("path/to/scripted_model.pt");

// Prepare input data
larcv::IOManager ioman(larcv::IOManager::kREAD);
ioman.add_in_file("merged_dlreco.root");
ioman.initialize();

// Create sparse image from larflow cluster
std::vector<std::vector<larpid::data::CropPixData_t>> sparse_images = 
    larpid::interface::make_prongCNN_input_sparse_images(
        ioman, prong_cluster, crop_center
    );

// Run inference
larpid::data::ModelOutput output = model.run_inference(sparse_images);

// Access results
std::cout << "Predicted PID: " << output.predictedPID << std::endl;
std::cout << "Purity: " << output.purity << std::endl;
std::cout << "Completeness: " << output.completeness << std::endl;
```

### Running the Main Executable

```bash
./run_larpid <merged_dlreco.root> <kpsreco.root> <model.pt> <output.root>
```

Arguments:
- `merged_dlreco.root`: DLGen2 merged file with wire and shower images
- `kpsreco.root`: KPS reconstruction file with neutrino candidates
- `model.pt`: Scripted PyTorch model file
- `output.root`: Output file for results

## Key Components

### TorchModel Class
Handles PyTorch model loading and inference:
- Loads scripted models via TorchScript
- Applies proper normalization (mean/std from training)
- Converts sparse pixel data to tensors
- Returns structured output with all four task predictions

### LArPIDInterface Functions
Provides image preparation utilities:
- `make_prongCNN_input_sparse_images()`: Main function for creating CNN input
- `getRecoImageBounds()`: Determines cropping boundaries
- `fillProngImages*()`: Fills prong pixel data with optional shower preservation
- `fillContextImages*()`: Adds surrounding context pixels

### Data Structures
- `CropPixData_t`: Stores sparse pixel information (row, col, value)
- `ModelOutput`: Contains all CNN predictions in organized format

## Model Details

The interface expects a TorchScript model with:
- **Input**: 6-channel 512×512 images (3 planes × 2 types: prong + context)
- **Output**: 4 heads for classification and regression tasks
- **Normalization**: Applied internally using training statistics

## Integration with DLGen2

LArPID is designed to work within the DLGen2 reconstruction chain:
1. Takes larflow clusters from track/shower reconstruction
2. Creates properly formatted sparse images
3. Runs inference to get particle properties
4. Results can be used for event selection and analysis

## Notes

- Models must be exported as TorchScript (`.pt` files)
- Minimum 10 pixels per plane required for inference
- At least 2 planes must have sufficient pixels
- Shower pixels can be optionally preserved in prong images
- Debug mode available for troubleshooting tensor operations

## Pre-trained Model

The default prongCNN model is available at:
`/uboone/data/users/mmr/prongCNN/ResNet34_recoProng_5class_epoch20.pt`