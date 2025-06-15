#include "ModelOutput.h"

namespace larpid {
namespace data {

ModelOutput::ModelOutput() {
    reset();
}

ModelOutput::~ModelOutput() {}

void ModelOutput::reset() {
    classScores.clear();
    predictedClass = -1;
    processScores.clear();
    predictedProcess = -1;
    completeness = 0.0;
    purity = 0.0;
}

} // namespace data
} // namespace larpid