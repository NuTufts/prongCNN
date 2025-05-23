#include "TorchModel.h"

namespace larpid {
namespace interface {

  //TorchModel::TorchModel(const std::string& model_path, const bool& useGPU){
  TorchModel::TorchModel(const std::string& model_path, const bool& debug){
  
    try {
      model = torch::jit::load(model_path);
      //device = (torch::cuda::is_available() && useGPU) ? torch::kCUDA : torch::kCPU;
      //model.to(device);
      model.to(torch::kCPU);
      model.eval();
    }
      catch (const c10::Error& e) {
        std::cerr << "LArPIDInterface: Error loading the torchscript model: " << e.what() << std::endl;
        throw;
    }
    
    debug_mode = debug;
    
    //norm_mean = torch::tensor({57.8182, 57.8182, 58.1807, 58.1807, 50.5312, 50.5312}, torch::kFloat32).view({1, 6, 1, 1}).to(device);
    //norm_std  = torch::tensor({62.9932, 62.9932, 62.6569, 62.6569, 42.0027, 42.0027}, torch::kFloat32).view({1, 6, 1, 1}).to(device);
    norm_mean = torch::tensor({57.8182, 57.8182, 58.1807, 58.1807, 50.5312, 50.5312}, torch::kFloat32).view({1, 6, 1, 1}).to(torch::kCPU);
    norm_std  = torch::tensor({62.9932, 62.9932, 62.6569, 62.6569, 42.0027, 42.0027}, torch::kFloat32).view({1, 6, 1, 1}).to(torch::kCPU);
  
  }


  TorchModel::TorchModel(){
  
    //norm_mean = torch::tensor({57.8182, 57.8182, 58.1807, 58.1807, 50.5312, 50.5312}, torch::kFloat32).view({1, 6, 1, 1}).to(device);
    //norm_std  = torch::tensor({62.9932, 62.9932, 62.6569, 62.6569, 42.0027, 42.0027}, torch::kFloat32).view({1, 6, 1, 1}).to(device);
    norm_mean = torch::tensor({57.8182, 57.8182, 58.1807, 58.1807, 50.5312, 50.5312}, torch::kFloat32).view({1, 6, 1, 1}).to(torch::kCPU);
    norm_std  = torch::tensor({62.9932, 62.9932, 62.6569, 62.6569, 42.0027, 42.0027}, torch::kFloat32).view({1, 6, 1, 1}).to(torch::kCPU);
  
  }


  void TorchModel::Initialize(const std::string& model_path, const bool& debug){
  
    try {
      model = torch::jit::load(model_path);
      //device = (torch::cuda::is_available() && useGPU) ? torch::kCUDA : torch::kCPU;
      //model.to(device);
      model.to(torch::kCPU);
      model.eval();
    }
    catch (const c10::Error& e) {
      std::cerr << "LArPIDInterface: Error loading the torchscript model: " << e.what() << std::endl;
      throw;
    }
    
    debug_mode = debug;
  
  }


  int TorchModel::getPID(const int& cnnClass){
    if(cnnClass == 0) return 11;
    if(cnnClass == 1) return 22;
    if(cnnClass == 2) return 13;
    if(cnnClass == 3) return 211;
    if(cnnClass == 4) return 2212;
    return 0;
  }


  void TorchModel::printTensorValues(const torch::Tensor& tensor) {
    auto accessor = tensor.accessor<float, 4>();
    for (int n = 0; n < accessor.size(0); ++n) {
      for (int c = 0; c < accessor.size(1); ++c) {
        for (int h = 0; h < accessor.size(2); ++h) {
          for (int w = 0; w < accessor.size(3); ++w) {
            std::cout << "tensor[" << n << "][" << c << "][" << h << "][" << w << "] = "
                      << accessor[n][c][h][w] << std::endl;
          }
        }
      }
    }
  }


  size_t TorchModel::getChannel(const size_t& pixDataIndex){
    if(pixDataIndex == 0) return 0; //plane0 prong
    if(pixDataIndex == 3) return 1; //plane0 context
    if(pixDataIndex == 1) return 2; //plane1 prong
    if(pixDataIndex == 4) return 3; //plane1 context
    if(pixDataIndex == 2) return 4; //plane2 prong
    if(pixDataIndex == 5) return 5; //plane2 context
    return std::numeric_limits<size_t>::max();
  }


  ModelOutput TorchModel::run_inference(const std::vector< std::vector<CropPixData_t> >& pixelData){

    torch::Tensor input_tensor = torch::zeros({1,6,512,512}, torch::kFloat32);
    //input_tensor = input_tensor.to(device);
    input_tensor = input_tensor.to(torch::kCPU);

    for(size_t v = 0; v < pixelData.size(); ++v){
      for(const auto& pix : pixelData[v]){
        if(pix.inCrop) input_tensor[0][getChannel(v)][pix.row][pix.col] = pix.val;
      }
    }

    input_tensor = (input_tensor - norm_mean) / norm_std;
    input_tensor = torch::clamp(input_tensor, -std::numeric_limits<float>::infinity(), 4.0);
    if(debug_mode) printTensorValues(input_tensor);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    auto output_tuple = model.forward(inputs).toTuple();
    at::Tensor outputT0 = output_tuple->elements()[0].toTensor();
    at::Tensor outputT1 = output_tuple->elements()[1].toTensor();
    at::Tensor outputT2 = output_tuple->elements()[2].toTensor();
    at::Tensor outputT3 = output_tuple->elements()[3].toTensor();

    ModelOutput output( getPID(outputT0.argmax(1).item<int>()), outputT3.argmax(1).item<int>(),
                        outputT1.item<float>(), outputT2.item<float>(), outputT0[0][0].item<float>(),
                        outputT0[0][1].item<float>(), outputT0[0][2].item<float>(),
                        outputT0[0][3].item<float>(), outputT0[0][4].item<float>(),
                        outputT3[0][0].item<float>(), outputT3[0][1].item<float>(),
                        outputT3[0][2].item<float>()
                      );

    return output;

  }

}
}