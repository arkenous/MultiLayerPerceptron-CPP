#include <iostream>
#include <vector>
#include "MultiLayerPerceptron.h"

/**
 * 与えられたデータを，0.0から1.0の間に収まるように正規化する
 * @param input 正規化する入力データ
 * @return 正規化後のデータ
 */
std::vector<double> normalize_between_zero_and_one(std::vector<double> input) {
    double xmax = 0, xmin = 0;
    for (int data = 0; data < input.size(); ++data) {
        if (xmax < input[data]) xmax = input[data];
        if (xmin > input[data]) xmin = input[data];
    }
    double xmax_minus_xmin = xmax - xmin;
    for (int data = 0; data < input.size(); ++data) {
        input[data] = (input[data] - xmin) / xmax_minus_xmin;
    }
    return input;
}

/**
 * データの加算平均が0.0，分散が1.0になるように正規化する
 * @param input 正規化する入力データ
 * @return 正規化後のデータ
 */
std::vector<double> normalize(std::vector<double> input) {
    // 一つのセットにおける平均値を求める
    double avg = 0;
    double sum = 0;
    for (int data = 0; data < input.size(); ++data) {
        sum += input[data];
    }
    avg = sum / input.size();
    // 偏差の二乗の総和を求める
    sum = 0;
    for (int data = 0; data < input.size(); ++data) {
        sum += std::pow((input[data] - avg), 2);
    }
    // 分散を求める
    double dispersion = sum / input.size();

    // 標準偏差を求める
    double standard_deviation = sqrt(dispersion);

    // 正規化し，得たデータで上書きする
    for (int data = 0; data < input.size(); ++data) {
        input[data] = (input[data] - avg) / standard_deviation;
    }

    return input;
}

int main() {
    std::vector<std::vector<double>> x = {
            {1.0, 1.0},
            {1.0, 0.0},
            {0.0, 1.0},
            {0.0, 0.0}
    };
    std::vector<std::vector<double>> answer = {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
    };

    MultiLayerPerceptron mlp = MultiLayerPerceptron(2, 2, 1, 1);
    mlp.learn(x, answer);

    mlp.out(x[0]);
    mlp.out(x[1]);
    mlp.out(x[2]);
    mlp.out(x[3]);

    return 0;
}

