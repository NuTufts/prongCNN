#include "TorchModel.h"
#include <iostream>
#include <stdexcept>

namespace larpid {
namespace interface {

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
        float mean_vals[] = {0.5924, 0.5924, 0.5924, 0.5924, 0.5924, 0.5924};
        float std_vals[] = {5.7890, 5.7890, 5.7890, 5.7890, 5.7890, 5.7890};
        
        norm_mean = torch::from_blob(mean_vals, {6}, torch::kFloat32);
        norm_std = torch::from_blob(std_vals, {6}, torch::kFloat32);
        
        if (debug_mode) {
            std::cout << "Model loaded from: " << model_path << std::endl;
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

ModelOutput TorchModel::run_inference(const std::vector<std::vector<CropPixData_t>>& pixelData) {
    ModelOutput output;
    
    try {
        // Create input tensor (1, 6, 512, 512)
        torch::Tensor input = torch::zeros({1, 6, 512, 512}, torch::kFloat32);
        
        // Fill tensor with pixel data
        for (size_t ch = 0; ch < pixelData.size() && ch < 6; ch++) {
            for (const auto& pix : pixelData[ch]) {
                if (pix.row >= 0 && pix.row < 512 && pix.col >= 0 && pix.col < 512) {
                    input[0][ch][pix.row][pix.col] = pix.adc;
                }
            }
        }
        
        // Normalize input
        input = (input - norm_mean.view({1, 6, 1, 1})) / norm_std.view({1, 6, 1, 1});
        
        if (debug_mode) {
            std::cout << "Input tensor prepared" << std::endl;
            printTensorValues(input);
        }
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        
        auto model_output = model.forward(inputs);
        
        // Parse output based on quadTask model structure
        if (model_output.isTuple()) {
            auto outputs = model_output.toTuple()->elements();
            
            // Classification output
            auto class_logits = outputs[0].toTensor();
            auto class_probs = torch::softmax(class_logits, 1);
            
            // Convert to vector
            auto class_probs_accessor = class_probs.accessor<float, 2>();
            output.classScores.resize(5);
            for (int i = 0; i < 5; i++) {
                output.classScores[i] = class_probs_accessor[0][i];
            }
            output.predictedClass = torch::argmax(class_logits, 1).item<int>();
            
            // Process classification output
            auto process_logits = outputs[1].toTensor();
            auto process_probs = torch::softmax(process_logits, 1);
            
            auto process_probs_accessor = process_probs.accessor<float, 2>();
            output.processScores.resize(3);
            for (int i = 0; i < 3; i++) {
                output.processScores[i] = process_probs_accessor[0][i];
            }
            output.predictedProcess = torch::argmax(process_logits, 1).item<int>();
            
            // Completeness regression
            auto completeness_tensor = outputs[2].toTensor();
            output.completeness = completeness_tensor[0][0].item<float>();
            
            // Purity regression
            auto purity_tensor = outputs[3].toTensor();
            output.purity = purity_tensor[0][0].item<float>();
        }
        
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