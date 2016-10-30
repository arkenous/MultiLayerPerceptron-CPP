//
// Created by Kensuke Kosaka on 2016/10/27.
//

#include <iostream>
#include <sstream>
#include "Neuron.h"

Neuron::Neuron(unsigned long inputNeuronNum) {
    this->inputNeuronNum = inputNeuronNum;
    this->inputWeights.reserve(this->inputNeuronNum);
    std::random_device rnd; // 非決定的乱数生成器
    std::mt19937 mt; // メルセンヌ・ツイスタ
    mt.seed(rnd());
    std::uniform_real_distribution<double> narrow_real_rnd(0.0, 1.0);
    this->threshold = narrow_real_rnd(mt); // 閾値を乱数で設定

    std::uniform_real_distribution<double> wide_real_rnd(0.0, 1.0);
    // 結合荷重をを乱数で初期化
    for (int i = 0; i < this->inputNeuronNum; ++i) {
        this->inputWeights.push_back(wide_real_rnd(mt));
    }
}

void Neuron::learn(double delta, std::vector<double> inputValues) {
    this->delta = delta;

    // 結合荷重の更新
    for (int i = 0; i < this->inputNeuronNum; ++i) {
        this->inputWeights[i] += this->eater * this->delta * inputValues[i];
    }
}

double Neuron::output(std::vector<double> inputValues){
    double sum = -this->threshold;
    for (int i = 0; i < this->inputNeuronNum; ++i) {
        sum += inputValues[i] * this->inputWeights[i];
    }

//    std::cout << "sum: " << sum <<std::endl;

    // 活性化関数を適用し，出力値を得る
//    std::cout << "activation(sum): " << activation_tanh(sum) << std::endl;

    return activation_tanh(sum);
}

double Neuron::activation_sigmoid(double x){
    return 1.0 / (1.0 + pow(M_E, -x));
}

double Neuron::activation_relu(double x) {
    return std::max(x, 0.0);
}

double Neuron::activation_tanh(double x) {
    return std::tanh(x);
}

double Neuron::getInputWeightIndexOf(int i){
    return this->inputWeights[i];
}

double Neuron::getDelta() {
    return this->delta;
}

std::string Neuron::toString() {
    std::cout << "weight[0]: " << inputWeights[0] << std::endl;
    std::stringstream ss;
    ss << "weight : ";
    for (int neuron = 0; neuron < inputNeuronNum; ++neuron) {
        ss << inputWeights[neuron] << " , ";
    }

    std::string output = ss.str();
    return output;
}
