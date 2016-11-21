
#include <random>
#include <sstream>
#include <iostream>
#include "StackedDenoisingAutoencoder.h"
#include "DenoisingAutoencoder.h"

using namespace std;

StackedDenoisingAutoencoder::StackedDenoisingAutoencoder() {}

std::string StackedDenoisingAutoencoder::learn(const std::vector<std::vector<double>> input,
                                               const unsigned long result_num_dimen, const float compression_rate) {
  std::stringstream ss;

  std::vector<std::vector<double>> answer(input);
  std::vector<std::vector<double>> noisy_input = add_noise(input, 0.1);
  DenoisingAutoencoder denoisingAutoencoder(noisy_input[0].size(), compression_rate);
  ss << denoisingAutoencoder.learn(answer, noisy_input) << "$";
  num_middle_neurons = denoisingAutoencoder.getCurrentMiddleNeuronNum();

  while (num_middle_neurons > result_num_dimen) {
    answer = std::vector<std::vector<double>>(noisy_input);
    noisy_input = add_noise(denoisingAutoencoder.getMiddleOutput(noisy_input), 0.1);
    denoisingAutoencoder = DenoisingAutoencoder(noisy_input[0].size(), compression_rate);
    ss << denoisingAutoencoder.learn(answer, noisy_input) << "$";
    num_middle_neurons = denoisingAutoencoder.getCurrentMiddleNeuronNum();
  }

  std::string result = ss.str();
  result.pop_back();
  ss.str("");
  ss.clear(stringstream::goodbit);

  num_middle_neurons = denoisingAutoencoder.getCurrentMiddleNeuronNum();

  return result;
}

/**
 * データごとに0.0以上1.0未満の乱数を生成し，rate未満であればそのデータを0.0にする
 * @param input ノイズをのせるデータ
 * @param rate ノイズをのせる確率
 * @return ノイズをのせたデータ
 */
std::vector<std::vector<double>> StackedDenoisingAutoencoder::add_noise(std::vector<std::vector<double>> input,
                                                                        float rate) {
  std::random_device rnd;
  std::mt19937 mt;
  mt.seed(rnd());
  std::uniform_real_distribution<double> real_rnd(0.0, 1.0);

  for (int i = 0; i < input.size(); ++i) {
    for (int j = 0; j < input[i].size(); ++j) {
      if (real_rnd(mt) < rate) input[i][j] = 0.0;
    }
  }

  return input;
}

unsigned long StackedDenoisingAutoencoder::getNumMiddleNeuron() {
  return num_middle_neurons;
}