//
// Created by Kensuke Kosaka on 2016/10/27.
//

#ifndef MULTILAYERPERCEPTRON_CPP_MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_CPP_MULTILAYERPERCEPTRON_H


#include <zconf.h>
#include "Neuron.h"

class MultiLayerPerceptron {
public:
    MultiLayerPerceptron(unsigned short input, unsigned short middle, unsigned short output, unsigned short middleLayer, int middleLayerType, double dropout_ratio);
    void learn(std::vector<std::vector<double>> x, std::vector<std::vector<double>> answer);
    std::string toString();
    double out(std::vector<double> input);
private:
    static const unsigned int MAX_TRIAL = 10000; // 学習上限回数
    constexpr static const double MAX_GAP = 0.1; // 許容する誤差の域値
    int num_thread = (int)sysconf(_SC_NPROCESSORS_ONLN); // プロセッサのコア数


    // ニューロン数
    unsigned short inputNumber = 0;
    unsigned short middleNumber = 0;
    unsigned short outputNumber = 0;

    unsigned short middleLayerNumber = 0; // 中間層の層数

    int middleLayerType = 0; // 中間層の活性化関数の種類指定．0: identity 1: sigmoid 2: tanh 3: ReLU

    bool successFlg = true;

    std::vector<std::vector<Neuron>> middleNeurons; // 中間層は複数層用意する
    std::vector<Neuron> outputNeurons;

    void outLearnThread(const std::vector<double> ans, const std::vector<double> o,
                        const std::vector<std::vector<double>> h, const int begin, const int end);
    void middleLastLayerLearnThread(const std::vector<std::vector<double>> h, const int begin, const int end);
    void middleMiddleLayerLearnThread(const std::vector<std::vector<double>> h, const int begin, const int end);
    void middleFirstLayerLearnThread(const std::vector<std::vector<double>> h, const std::vector<double> in, const int begin, const int end);
};


#endif //MULTILAYERPERCEPTRON_CPP_MULTILAYERPERCEPTRON_H
