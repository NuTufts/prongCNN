#ifndef __LARPID_INTERFACE_TORCH_MODEL_H__
#define __LARPID_INTERFACE_TORCH_MODEL_H__


// hide this from the ROOT interpretter
#ifndef __CINT__
#ifndef __CLING__

//ClassDef macro from ROOT and libtorch conflict, this is necessary:
#ifdef ClassDef
#undef ClassDef
#endif
#include <torch/script.h>

//Load ROOT ClassDef back in now that we have the torch headers:
#ifdef ClassDef
#undef ClassDef
#endif
#include <Rtypes.h>

#include <vector>
#include <string>

#include "larpid/interface/ModelOutput.h"
#include "larpid/interface/CropPixData_t.h"

namespace larpid {
namespace interface {

  class TorchModel {

  private:
    torch::jit::Module model;
    //torch::Device device;
    torch::Tensor norm_mean;
    torch::Tensor norm_std;
    bool debug_mode;
    int getPID(const int& cnnClass);
    size_t getChannel(const size_t& pixDataIndex);
    void printTensorValues(const torch::Tensor& tensor);

  public:
    //TorchModel(const std::string& model_path, const bool& useGPU=false);
    TorchModel(); //must call Initialize before using model if using this constructor
    TorchModel(const std::string& model_path, const bool& debug=false);
    void Initialize(const std::string& model_path, const bool& debug=false);
    ModelOutput run_inference(const std::vector< std::vector<CropPixData_t> >& pixelData);

  };

}
}

#endif
#endif

#endif