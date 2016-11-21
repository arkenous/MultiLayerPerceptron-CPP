
#ifndef MULTILAYERPERCEPTRON_CPP_STACKEDDENOISINGAUTOENCODER_H
#define MULTILAYERPERCEPTRON_CPP_STACKEDDENOISINGAUTOENCODER_H

#include <vector>
#include <string>

class StackedDenoisingAutoencoder {
public:
  StackedDenoisingAutoencoder();

  std::string learn(const std::vector<std::vector<double>> input, const unsigned long result_num_dimen,
                    const float compression_rate);

  unsigned long getNumMiddleNeuron();

private:
  std::vector<std::vector<double>> add_noise(std::vector<std::vector<double>> input, float rate);

  unsigned long num_middle_neurons;
};

#endif //MULTILAYERPERCEPTRON_CPP_STACKEDDENOISINGAUTOENCODER_H
