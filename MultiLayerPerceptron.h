//
// Created by Kensuke Kosaka on 2016/10/27.
//

#ifndef MULTILAYERPERCEPTRON_CPP_MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_CPP_MULTILAYERPERCEPTRON_H


#include <zconf.h>
#include "Neuron.h"

class MultiLayerPerceptron {
public:
  MultiLayerPerceptron(unsigned long input, unsigned long middle, unsigned long output, unsigned long middleLayer,
                       int middleLayerType, double dropout_rate, std::string sda_params);

  void learn(std::vector<std::vector<double>> x, std::vector<std::vector<double>> answer);

  std::string toString();

  std::vector<double> out(std::vector<double> input, bool showResult);

private:
  static const unsigned int MAX_TRIAL = 100000; // 学習上限回数
  constexpr static const double MAX_GAP = 0.1; // 許容する誤差の域値
  int num_thread = (int) sysconf(_SC_NPROCESSORS_ONLN); // プロセッサのコア数


  // ニューロン数
  unsigned long inputNumber = 0;
  unsigned long middleNumber = 0;
  unsigned long outputNumber = 0;

  unsigned long middleLayerNumber = 0; // 中間層の層数

  int middleLayerType = 0; // 中間層の活性化関数の種類指定．0: identity 1: sigmoid 2: tanh 3: ReLU

  bool successFlg = true;

  std::vector<std::vector<Neuron>> sda_neurons;
  std::vector<std::vector<double>> sda_out;

  std::vector<std::vector<Neuron>> middleNeurons; // 中間層は複数層用意する
  std::vector<Neuron> outputNeurons;
  std::vector<std::vector<double>> h;
  std::vector<double> o;

  std::vector<std::vector<double>> learnedH;
  std::vector<double> learnedO;

  std::vector<std::vector<Neuron>> setupSdA(std::string sda_params);

  std::vector<double> separate_by_camma(std::string input);

  void sdaFirstLayerOutThread(const std::vector<double> in, const int begin, const int end);

  void sdaOtherLayerOutThread(const int layer, const int begin, const int end);

  void middleFirstLayerForwardThread(const int begin, const int end);

  void middleLayerForwardThread(const int layer, const int begin, const int end);

  void outForwardThread(const int begin, const int end);

  void outLearnThread(const std::vector<double> in, const std::vector<double> ans, const int begin, const int end);

  void middleLastLayerLearnThread(const int begin, const int end);

  void middleMiddleLayerLearnThread(const int layer, const int begin, const int end);

  void middleFirstLayerLearnThread(const int begin, const int end);

  void middleFirstLayerOutThread(const int begin, const int end);

  void middleLayerOutThread(const int layer, const int begin, const int end);

  void outOutThread(const int begin, const int end);
};


#endif //MULTILAYERPERCEPTRON_CPP_MULTILAYERPERCEPTRON_H
