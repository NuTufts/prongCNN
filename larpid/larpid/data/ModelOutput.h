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
    std::vector<float> classScores; // the logSoftmax is stored
    int predictedClass; ///< predicted class CNN index
    int predictedPID;   ///< PDG code of predicted class
    
    // Process classification results
    std::vector<float> processScores; // the logSoftmax is stored
    int predictedProcess;
    
    // Regression results
    float completeness; // the output of sigmoid is stored
    float purity;       // the output of sigmoid is stored
    
    // Reset all values
    void reset();
  };

} // namespace interface
} // namespace larpid

#endif