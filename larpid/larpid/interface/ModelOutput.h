#ifndef __LARPID_INTERFACE_MODELOUTPUT_H__
#define __LARPID_INTERFACE_MODELOUTPUT_H__

#include <vector>
#include <string>

namespace larpid {
namespace interface {

  class ModelOutput {
  public:
    ModelOutput();
    ~ModelOutput();
    
    // Classification results
    std::vector<float> classScores;
    int predictedClass;
    
    // Process classification results
    std::vector<float> processScores;
    int predictedProcess;
    
    // Regression results
    float completeness;
    float purity;
    
    // Reset all values
    void reset();
  };

} // namespace interface
} // namespace larpid

#endif