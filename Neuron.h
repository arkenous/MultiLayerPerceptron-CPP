//
// Created by Kensuke Kosaka on 2016/10/27.
//

#ifndef MULTILAYERPERCEPTRON_CPP_NEURON_H
#define MULTILAYERPERCEPTRON_CPP_NEURON_H


#include <vector>
#include <random>

class Neuron {
public:
    Neuron(unsigned long inputNeuronNum, int activation_type);
    void learn(double delta, std::vector<double> inputValues);
    double output(std::vector<double> inputValues);
    double getInputWeightIndexOf(int i);
    double getDelta();
    std::string toString();
private:
    unsigned long inputNeuronNum = 0;
    int activation_type = 0;
    std::vector<double> inputWeights;
    double delta = 0.0; // 修正量
    double bias = 0.0; // ニューロンのバイアス // -threshold
    double alpha = 0.01; // 学習率，AdaGradで学習率を更新する
    std::vector<double> g; // 学習率用AdaGrad．過去の勾配の二乗和を覚えておく
    double epsilon = 0.00000001;
    double rambda = 0.00001; // 荷重減衰の定数．正の小さな定数にしておくことで勾配がゼロでも重みが減る
    double activation_identity(double x); // 0
    double activation_sigmoid(double x); // 1
    double activation_tanh(double x); // 2
    double activation_relu(double x); // 3
};

#endif //MULTILAYERPERCEPTRON_CPP_NEURON_H
