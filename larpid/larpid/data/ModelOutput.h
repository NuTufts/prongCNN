#ifndef __LARPID_DATA_MODELOUTPUT_H__
#define __LARPID_DATA_MODELOUTPUT_H__

#include <vector>

namespace larpid {
namespace data {

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