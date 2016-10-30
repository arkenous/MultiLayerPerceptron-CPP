//
// Created by Kensuke Kosaka on 2016/10/27.
//

#ifndef MULTILAYERPERCEPTRON_CPP_NEURON_H
#define MULTILAYERPERCEPTRON_CPP_NEURON_H


#include <vector>
#include <random>

class Neuron {
public:
    Neuron(unsigned long inputNeuronNum);
    void learn(double delta, std::vector<double> inputValues);
    double output(std::vector<double> inputValues);
    double getInputWeightIndexOf(int i);
    double getDelta();
    std::string toString();
private:
    unsigned long inputNeuronNum = 0;
    std::vector<double> inputWeights;
    double delta = 0.0;
    double threshold = 0.0;
    double eater = 0.3;
    double activation_sigmoid(double x);
    double activation_relu(double x);
    double activation_tanh(double x);
};


#endif //MULTILAYERPERCEPTRON_CPP_NEURON_H
