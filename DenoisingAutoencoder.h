
#ifndef MULTILAYERPERCEPTRON_CPP_DENOISINGAUTOENCODER_H
#define MULTILAYERPERCEPTRON_CPP_DENOISINGAUTOENCODER_H

#include <string>
#include <vector>
#include <zconf.h>
#include "Neuron.h"

class DenoisingAutoencoder {
public:
  DenoisingAutoencoder(unsigned long num_input, float compression_rate);

  std::string learn(std::vector<std::vector<double>> input, std::vector<std::vector<double>> noisy_input);

  std::vector<double> out(std::vector<double> input, bool showResult);

  std::vector<std::vector<double>> getMiddleOutput(std::vector<std::vector<double>> noisy_input);

  unsigned long getCurrentMiddleNeuronNum();

private:
  static const unsigned int MAX_TRIAL = 50; // 学習上限回数
  constexpr static const double MAX_GAP = 1.0; // 許容する誤差
  int num_thread = (int) sysconf(_SC_NPROCESSORS_ONLN);

  unsigned long input_neuron_num;
  unsigned long middle_neuron_num;
  unsigned long output_neuron_num;

  int middle_layer_type = 0; // 中間層の活性化関数の種類指定：0: identity, 1: sigmoid, 2: tanh, 3: ReLU

  bool successFlg = true;

  std::vector<Neuron> middle_neurons;
  std::vector<Neuron> output_neurons;

  std::vector<double> h; // 中間層の出力値
  std::vector<double> o; // 出力層の出力値
  std::vector<double> learnedH;
  std::vector<double> learnedO;


  void middleForwardThread(const std::vector<double> in, const int begin, const int end);

  void outForwardThread(const int begin, const int end);

  void outLearnThread(const std::vector<double> in, const std::vector<double> ans, const int begin, const int end);

  void middleLearnThread(const std::vector<double> in, const int begin, const int end);

  void middleOutThread(const std::vector<double> in, const int begin, const int end);

  void outOutThread(const int begin, const int end);
};

#endif //MULTILAYERPERCEPTRON_CPP_DENOISINGAUTOENCODER_H
