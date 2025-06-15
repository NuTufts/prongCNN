#include <iostream>

#include  "larpid/model/TorchModel.h"

int main( int nargs, char** argv ) {

  std::cout << "Test Load Model: provide weights and load LArPID using libtorch" << std::endl;

  larpid::model::TorchModel model;

  std::string model_file = argv[1];

  std::cout << "loading model file: " << model_file << std::endl;

  bool debug = true;

  model.Initialize( model_file, debug );

  return 0;
  
}
