#ifndef __LARPID_INTERFACE_TORCH_MODEL_H__
#define __LARPID_INTERFACE_TORCH_MODEL_H__

#ifndef __CINT__
#ifndef __CLING__

#include <torch/script.h>

#include "larpid/data/ModelOutput.h"
#include "larpid/data/CropPixData_t.h"

namespace larpid {
namespace model {

  class TorchModel {

  private:
    torch::jit::script::Module model;
    //torch::jit::Module model;
    //torch::Device device;
    torch::Tensor norm_mean;
    torch::Tensor norm_std;
    bool debug_mode;
    int getPID(const int& cnnClass);
    size_t getChannel(const size_t& pixDataIndex);
    void printTensorValues(const torch::Tensor& tensor);

    std::vector<float> _mean_vals;
    std::vector<float> _std_vals;

  public:
    //TorchModel(const std::string& model_path, const bool& useGPU=false);
    TorchModel(); //must call Initialize before using model if using this constructor
    TorchModel(const std::string& model_path, const bool& debug=false);
    void Initialize(const std::string& model_path, const bool& debug=false);
    larpid::data::ModelOutput run_inference(const std::vector< std::vector<larpid::data::CropPixData_t> >& pixelData);

  };

}
}

#endif // ifndef CLING
#endif // ifndef CINT

#endif