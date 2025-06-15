#include "TorchModel.h"
#include <iostream>
#include <stdexcept>

#include <torch/torch.h>
#include <c10/util/TypeIndex.h>

namespace larpid {
namespace model {

TorchModel::TorchModel() : debug_mode(false) {
    // Default constructor - model must be initialized later
}

TorchModel::TorchModel(const std::string& model_path, const bool& debug) 
    : debug_mode(debug) {
    Initialize(model_path, debug);
}

void TorchModel::Initialize(const std::string& model_path, const bool& debug) {
    debug_mode = debug;
    
    try {
        // Load the model
        model = torch::jit::load(model_path);
        model.eval();
        
        // Initialize normalization parameters (from dataset mean/std)
        // These values should match those in datasets_reco_5ClassHardLabel_quadTask.py
        _mean_vals = std::vector<float>{0.5924, 0.5924, 0.5924, 0.5924, 0.5924, 0.5924};
        _std_vals  = std::vector<float>{5.7890, 5.7890, 5.7890, 5.7890, 5.7890, 5.7890};
        
        norm_mean = torch::from_blob(_mean_vals.data(), {1,6,1,1}, torch::kFloat32);
        norm_std  = torch::from_blob(_std_vals.data(),  {1,6,1,1}, torch::kFloat32);

        // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        // norm_mean = torch::zeros({1,6,1,1}, options);
        // norm_std  = torch::zeros({1,6,1,1}, options);
        // auto acc_norm_mean = norm_mean.accessor<float, 4>();
        // auto acc_norm_std  = norm_std.accessor<float, 4>();
        // for (int i=0; i<6; i++) {
        //     acc_norm_mean[0][i][0][0] = _mean_vals[i];
        //     acc_norm_std[0][i][0][0]  = _std_vals[i];
        // }
        
        if (debug_mode) {
            std::cout << "Model loaded from: " << model_path << std::endl;

            std::cout << "Made norm_mean tensor: " << std::endl;
            printTensorValues( norm_mean );

            std::cout << "Made norm_std tensor: " << std::endl;
            printTensorValues( norm_std );
        }
    }
    catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}

int TorchModel::getPID(const int& cnnClass) {
    // Convert CNN class to PDG code
    switch(cnnClass) {
        case 0: return 11;    // electron
        case 1: return 22;    // photon
        case 2: return 13;    // muon
        case 3: return 211;   // pion
        case 4: return 2212;  // proton
        default: return 0;
    }
}

size_t TorchModel::getChannel(const size_t& pixDataIndex) {
    // Map pixel data index to channel
    // Assuming 3 planes × 2 image types = 6 channels
    return pixDataIndex;
}

void TorchModel::printTensorValues(const torch::Tensor& tensor) {
    if (!debug_mode) return;
    
    std::cout << "Tensor shape: " << tensor.sizes() << std::endl;
    std::cout << "Tensor values: " << tensor << std::endl;
}

larpid::data::ModelOutput 
TorchModel::run_inference(const std::vector<std::vector<larpid::data::CropPixData_t>>& pixelData) {
    
    larpid::data::ModelOutput output;
    
    try {
        // Create input tensor (1, 6, 512, 512)
        torch::Tensor input = torch::zeros({1, 6, 512, 512}, torch::kFloat32);
        
        // Fill tensor with pixel data
        for (size_t ch = 0; ch < pixelData.size() && ch < 6; ch++) {
            for (const auto& pix : pixelData[ch]) {
                if (pix.row >= 0 && pix.row < 512 && pix.col >= 0 && pix.col < 512) {

                    if ( std::isnan(pix.adc) || std::isinf(pix.adc) ) {
                        std::cerr << "Bad pixel value "
                                  << "@ [" << ch << ", " << pix.row << ", " << pix.col << "] "
                                  << " pix.adc=" << pix.adc << std::endl;
                        throw std::runtime_error("Bad input tensor pixel value");
                    }

                    input[0][ch][pix.row][pix.col] = pix.adc;
                }
            }
        }
        
        torch::Tensor is_inf_prenorm_tensor = torch::isinf(input); 

        if (is_inf_prenorm_tensor.any().item<bool>()) {
            std::cerr << "Input tensor (pre-normalization) has inf values" << std::endl;
            throw std::runtime_error("Input tensor (pre-normalized) has inf values");
            return output;
        }
        else {
            if ( debug_mode )
                std::cout << "Prenorm tensor is good: does not have inf values" << std::endl;
        }

        if ( debug_mode ) {
            std::cout << "View norm and std tensors" << std::endl;
            printTensorValues( norm_mean.view({1, 6, 1, 1}) );
            printTensorValues( norm_std.view({1, 6, 1, 1}) );
        }

        // Normalize input
        input = (input - norm_mean.view({1, 6, 1, 1})) / norm_std.view({1, 6, 1, 1});
        
        torch::Tensor is_inf_tensor = torch::isinf(input); 

        if (is_inf_tensor.any().item<bool>()) {
            std::cerr << "Input tensor (after normalization) has inf values" << std::endl;
            throw std::runtime_error("Input tensor (after normalization) has inf values");
            return output;
        }
        else {
            if ( debug_mode )
                std::cout << "Input tensor (after normalization) is good: no inf values" << std::endl;
        }

        if (debug_mode) {
            std::cout << "Input tensor prepared" << std::endl;
            //printTensorValues(input);
        }

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        
        auto model_output = model.forward(inputs);

        if (debug_mode) {
            std::cout << "model_output.tag()=" << model_output.tagKind() << std::endl;
        }
        
        // Parse output based on quadTask model structure
        torch::Tensor class_probs;
        torch::Tensor process_probs;
        torch::Tensor completeness_tensor;
        torch::Tensor purity_tensor;

        if ( model_output.isList() ) {
            auto outputs = model_output.toList();
            class_probs = outputs.get(0).toTensor();
            completeness_tensor = outputs.get(1).toTensor();
            purity_tensor = outputs.get(2).toTensor();
            process_probs = outputs.get(3).toTensor();
        }
        else if ( model_output.isTuple() ) {
            auto outputs = model_output.toTuple()->elements();
            class_probs = outputs[0].toTensor();
            completeness_tensor = outputs[1].toTensor();
            purity_tensor = outputs[2].toTensor();
            process_probs = outputs[3].toTensor();
        }
        else {
            std::stringstream ss;
            ss << "Model output tag unknown: " << model_output.tagKind() << std::endl;
            throw std::runtime_error(ss.str());
        }
        
        if ( debug_mode) {
            std::cout << "Class logits" << std::endl;
            printTensorValues( class_probs );
            std::cout << "Class probs (after softmax): " << std::endl;
            printTensorValues( class_probs );
            std::cout << "Completeness tensor:" << std::endl;
            printTensorValues( completeness_tensor );
            std::cout << "Purity tensor: " << std::endl;
            printTensorValues( purity_tensor );
            std::cout << "Process logits: " << std::endl;
            printTensorValues( process_probs );
        }
        
        // Convert to vector
        if ( debug_mode ) std::cout << "store classification scores" << std::endl;
        auto class_probs_accessor = class_probs.accessor<float, 2>();
        output.classScores.resize(5);
        for (int i = 0; i < 5; i++) {
            output.classScores[i] = class_probs_accessor[0][i];
        }
        output.predictedClass = torch::argmax(class_probs, 1).item<int>();
        
        // Process classification output
        if ( debug_mode ) std::cout << "store process scores" << std::endl;
        auto process_probs_accessor = process_probs.accessor<float, 2>();
        output.processScores.resize(3);
        for (int i = 0; i < 3; i++) {
            output.processScores[i] = process_probs_accessor[0][i];
        }
        output.predictedProcess = torch::argmax(process_probs, 1).item<int>();
            
        // Completeness regression
        if ( debug_mode ) std::cout << "store completeness score" << std::endl;
        output.completeness = completeness_tensor.item<float>();
        
        // Purity regression
        if ( debug_mode ) std::cout << "store purity score" << std::endl;
        output.purity = purity_tensor.item<float>();
        
        if (debug_mode) {
            std::cout << "Inference complete. Predicted class: " << output.predictedClass 
                     << " (PDG: " << getPID(output.predictedClass) << ")" << std::endl;
        }
    }
    catch (const c10::Error& e) {
        throw std::runtime_error("Error during inference: " + std::string(e.what()));
    }
    
    return output;
}

} // namespace interface
} // namespace larpid