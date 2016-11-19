#include <iostream>
#include <vector>
#include "MultiLayerPerceptron.h"
#include "Data.h"

/**
 * データの加算平均が0.0，分散が1.0になるように正規化する
 * @param input 正規化する入力データ
 * @return 正規化後のデータ
 */
std::vector<double> normalize(std::vector<double> input) {
    // 一つのセットにおける平均値を求める
    double sum = 0.0;
    for (int data = 0; data < input.size(); ++data) {
        sum += input[data];
    }
    double avg = sum / input.size();
    // 偏差の二乗の総和を求める
    sum = 0.0;
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
    double numSucceed = 0.0;
    double dropout_ratio = 0.5;
    for (int loop = 0; loop < 100; ++loop) {

        for (int i = 0; i < train.size(); ++i) {
            train[i] = normalize(train[i]);
        }
        for (int i = 0; i < test_success.size(); ++i) {
            test_success[i] = normalize(test_success[i]);
        }
        for (int i = 0; i < test_fail.size(); ++i) {
            test_fail[i] = normalize(test_fail[i]);
        }

        MultiLayerPerceptron mlp = MultiLayerPerceptron((unsigned short) train[0].size(),
                                                        (unsigned short) train[0].size(),
                                                        (unsigned short) answer[0].size(), 1, 1, dropout_ratio);
        mlp.learn(train, answer);

        std::cout << "--- NaN check ---" << std::endl;
        while (isnan(mlp.out(train[0], true)[0])) {
            std::cout << "is NaN" << std::endl;
            mlp = MultiLayerPerceptron((unsigned short) train[0].size(), (unsigned short) train[0].size(), (unsigned short) answer[0].size(), 1, 1, dropout_ratio);
            mlp.learn(train, answer);
        }

        std::cout << "----------     Success     ----------" << std::endl;
        for (int i = 0; i < test_success.size(); ++i) {
            if (mlp.out(test_success[i], true)[0] < 0.5) numSucceed += 1.0;
        }
        std::cout << "----------     Fail     ----------" << std::endl;
        for (int i = 0; i < test_fail.size(); ++i) {
            if (mlp.out(test_fail[i], true)[0] >= 0.5) numSucceed += 1.0;
        }

    }

    double percentage = (numSucceed / 1000.0) * 100.0;
    std::cout << "正答率: " << percentage << "%" << std::endl;
    return 0;
}

