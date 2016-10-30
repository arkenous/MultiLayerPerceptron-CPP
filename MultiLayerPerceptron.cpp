//
// Created by Kensuke Kosaka on 2016/10/27.
//

#include <mach/boolean.h>
#include "MultiLayerPerceptron.h"
#include "iostream"
#include <thread>

MultiLayerPerceptron::MultiLayerPerceptron(unsigned short input, unsigned short middle, unsigned short output, unsigned short middleLayer) {
    this->inputNumber = input;
    this->middleNumber = middle;
    this->outputNumber = output;
    this->middleLayerNumber = middleLayer;
    this->middleNeurons.reserve(middleLayerNumber);
    for (int i = 0; i < middleLayerNumber; ++i) {
        this->middleNeurons[i].reserve(middleNumber);
    }
    this->outputNeurons.reserve(outputNumber);

    for (int neuron = 0; neuron < output; ++neuron) {
        this->outputNeurons.push_back(Neuron(inputNumber));
    }

    std::vector<Neuron> neuronPerLayer;
    neuronPerLayer.reserve(middleNumber);

    for (int layer = 0; layer < middleLayerNumber; ++layer) {
        for (int neuron = 0; neuron < middleNumber; ++neuron) {
            neuronPerLayer.push_back(Neuron(inputNumber));
        }
        this->middleNeurons.push_back(neuronPerLayer);
        neuronPerLayer.clear();
    }
}

void MultiLayerPerceptron::learn(std::vector<std::vector<double>> x, std::vector<std::vector<double>> answer) {
    std::vector<std::vector<double>> h = std::vector<std::vector<double>>(middleLayerNumber, std::vector<double>(middleNumber, 0));
    std::vector<double> o = std::vector<double>(outputNumber, 0);

    int succeed = 0; //  連続正解回数のカウンタを初期化
    for (int trial = 0; trial < this->MAX_TRIAL; ++trial) {
        std::cout << std::endl;
        std::cout << "Trial:" << trial << std::endl;

        // 使用する教師データを選択
        std::vector<double> in = x[trial % answer.size()]; // 利用する教師入力データ
        std::vector<double> ans = answer[trial % answer.size()]; // 教師出力データ

        // 出力値を推定：1層目の中間層の出力計算
        for (int neuron = 0; neuron < this->middleNumber; ++neuron) {
            h[0][neuron] = middleNeurons[0][neuron].output(in);
//            std::cout << "h[0][" << neuron << "]: " << h[0][neuron] << std::endl;
        }

        // 一つ前の中間層より得られた出力を用いて，以降の中間層を順に計算
        for (int layer = 1; layer < middleLayerNumber; ++layer) {
            for (int neuron = 0; neuron < middleNumber; ++neuron) {
                h[layer][neuron] = middleNeurons[layer][neuron].output(h[layer - 1]);
//                std::cout << "h[" << layer << "][" << neuron << "]: " << h[layer][neuron] << std::endl;
            }
        }

        // 出力値を推定：中間層の最終層の出力を用いて，出力層の出力計算
        for (int neuron = 0; neuron < outputNumber; ++neuron) {
            o[neuron] = outputNeurons[neuron].output(h[middleLayerNumber - 1]);
//            std::cout << "before o[" << neuron << "]: " << o[neuron] << std::endl;
        }

        /*
        // 教師入力データを出力
        std::cout << "[input] ";
        for (int i = 0; i < in.size(); ++i) {
          std::cout << in[i] << " ";
        }
        std::cout << std::endl;

        // 教師出力データを出力
        std::cout << "[answer] " << ans << std::endl;

        // 出力層から得られたデータを出力
        std::cout << "[output] " << o[0] << std::endl;

        // 中間層から得られたデータを出力
        for (int layer = 0; layer < middleLayerNumber; ++layer) {
          std::cout << "[middle " << layer << "] ";
          for (int neuron = 0; neuron < middleNumber; ++neuron) {
            std::cout << h[layer][neuron] << " ";
          }
          std::cout << std::endl;
        }
        */

        successFlg = true;

        //region 出力層を学習する
        //TODO ニューロンの組を半分ずつ分担するとかで並列化できそう
        std::vector<std::thread> threads;
        int charge = 1;
        if (outputNumber <= num_thread) charge = 1;
        else charge = outputNumber / num_thread;
        for (int i = 0; i < outputNumber; i += charge) {
            if (i != 0 && outputNumber / i == 1) {
//                std::cout << "i: " << i << "   outputNumber: " << outputNumber << std::endl;
                threads.push_back(std::thread(&MultiLayerPerceptron::outLearnThread, this, std::ref(ans), std::ref(o), std::ref(h), i, outputNumber));
            } else {
//                std::cout << "i: " << i << "   i + charge: " << i + charge << std::endl;
                threads.push_back(std::thread(&MultiLayerPerceptron::outLearnThread, this, std::ref(ans), std::ref(o), std::ref(h), i, i + charge));
            }
        }
        for (std::thread &th : threads) th.join();
        //endregion

        // 連続成功回数による終了判定
        if (successFlg) {
            succeed++;
            if (succeed >= x.size()) break;
            else continue;
        } else succeed = 0;

        //region 中間層の更新．末尾層から先頭層に向けて更新する

        //region 中間層の層数が2以上の場合のみ，中間層の最終層の学習をする
        if (middleLayerNumber > 1) {
            threads.clear();
            //TODO 出力層同様，ここも並列化できそう
            if (middleNumber <= num_thread) charge = 1;
            else charge = middleNumber / num_thread;
            for (int i = 0; i < middleNumber; i += charge) {
                if (i != 0 && middleNumber / i == 1) {
                    threads.push_back(std::thread(&MultiLayerPerceptron::middleLastLayerLearnThread, this, std::ref(h), i, middleNumber));
                } else {
                    threads.push_back(std::thread(&MultiLayerPerceptron::middleLastLayerLearnThread, this, std::ref(h), i, i + charge));
                }
            }
            for (std::thread &th : threads) th.join();
        }
        //endregion

        //region 出力層と入力層に最も近い層一つずつを除いた残りの中間層を入力層に向けて学習する
        threads.clear();
        if (middleNumber <= num_thread) charge = 1;
        else charge = middleNumber / num_thread;
        for (int i = 0; i < middleNumber; i += charge) {
            if (i != 0 && middleNumber / i == 1) {
                threads.push_back(std::thread(&MultiLayerPerceptron::middleMiddleLayerLearnThread, this, std::ref(h), i, middleNumber));
            } else {
                threads.push_back(std::thread(&MultiLayerPerceptron::middleMiddleLayerLearnThread, this, std::ref(h), i, i + charge));
            }
        }
        for (std::thread &th : threads) th.join();
        //endregion

        //region 中間層の最初の層を学習する
        threads.clear();
        if (middleNumber <= num_thread) charge = 1;
        else charge = middleNumber / num_thread;
        for (int i = 0; i < middleNumber; i += charge) {
            if (i != 0 && middleNumber / i == 1) {
                threads.push_back(std::thread(&MultiLayerPerceptron::middleFirstLayerLearnThread, this, std::ref(h), std::ref(in), i, middleNumber));
            } else {
                threads.push_back(std::thread(&MultiLayerPerceptron::middleFirstLayerLearnThread, this, std::ref(h), std::ref(in), i, i + charge));
            }
        }
        for (std::thread &th : threads) th.join();
        //endregion

        //endregion

        // 再度出力
        // 出力値を推定：中間層の出力計算
        // 1層目の中間層の出力を計算
        for (int neuron = 0; neuron < middleNumber; ++neuron) {
            h[0][neuron] = middleNeurons[0][neuron].output(in);
//            std::cout << "h[0][" << neuron << "]: " << h[0][neuron] << std::endl;
        }

        // 一つ前の中間層より得られた出力を用いて，以降の中間層を順に計算
        for (int layer = 1; layer < middleLayerNumber; ++layer) {
            for (int neuron = 0; neuron < middleNumber; ++neuron) {
                h[layer][neuron] = middleNeurons[layer][neuron].output(h[layer - 1]);
//                std::cout << "h[" << layer << "][" << neuron << "]: " << h[layer][neuron] << std::endl;
            }
        }

        // 出力値を推定：出力層の出力計算
        // 中間層の最終層の出力を用いて，出力層の出力を計算
        for (int neuron = 0; neuron < outputNumber; ++neuron) {
            o[neuron] = outputNeurons[neuron].output(h[middleLayerNumber - 1]);
//            std::cout << "after o[" << neuron << "]: " << o[neuron] << std::endl;
        }

        /*
        std::cout << "[input] ";
        for (int i = 0; i < in.size(); ++i) {
          std::cout << in[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "[output] " << o[0] << std::endl;

        for (int layer = 0; layer < middleLayerNumber; ++layer) {
          std::cout << "[middle " << layer << "] ";
          for (int i = 0; i < h[layer].size(); ++i) {
            std::cout << h[layer][i] << " ";
          }
          std::cout << std::endl;
        }
        */
    }

    // 全ての教師データで正解を出すか，収束限度回数を超えた場合に終了
//    std::cout << "[finish] " << this->toString() << std::endl;
    std::cout << "[finish]" << std::endl;
}

std::string MultiLayerPerceptron::toString() {
    // 戻り値変数
    std::string str = "";

    // 中間層ニューロン出力
    str += " middle neurons ( ";
    for (int layer = 0; layer < middleLayerNumber; ++layer) {
        for (int neuron = 0; neuron < middleNumber; ++neuron) {
            str += middleNeurons[layer][neuron].toString();
        }
    }
    str += ") ";

    // 出力層ニューロン出力
    str += " output neurons ( ";
    for (int neuron = 0; neuron < outputNumber; ++neuron) {
        str += outputNeurons[neuron].toString();
    }
    str += ") ";

    return str;
}

void MultiLayerPerceptron::outLearnThread(const std::vector<double> ans, const std::vector<double> o,
                                          const std::vector<std::vector<double>> h, const int begin, const int end){
    for (int neuron = begin; neuron < end; ++neuron) {
        // 出力層ニューロンの学習係数deltaを計算
        double delta = (ans[neuron] - o[neuron]) * o[neuron] * (1.0 - o[neuron]);

        // 教師データとの誤差が十分小さい場合は学習しない．そうでなければ正解フラグをfalseに
//        std::cout << "delta: " << delta << "= (" << ans[neuron] << " - " << o[neuron] << ") * " << o[neuron] << " * (1.0 - " << o[neuron] << ")" << std::endl;
        std::cout << "GAP: " << std::abs(ans[neuron] - o[neuron]) << std::endl;
        if (std::abs(ans[neuron] - o[neuron]) < MAX_GAP) continue;
        else successFlg = false;

        // 出力層の学習
//        std::cout << "[learn] before o: " << outputNeurons[neuron].toString() << std::endl;
        outputNeurons[neuron].learn(delta, h[middleLayerNumber - 1]);
//        std::cout << "[learn] after o: " << outputNeurons[neuron].toString() << std::endl;
    }
}

void MultiLayerPerceptron::middleLastLayerLearnThread(const std::vector<std::vector<double>> h, const int begin,
                                                      const int end){
    for (int neuron = begin; neuron < end; ++neuron) {
        // 中間層ニューロンの学習係数deltaを計算
        double sumDelta = 0.0;
        for (int k = 0; k < outputNumber; ++k) {
            Neuron n = outputNeurons[k];
            sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
        }
        double delta = h[middleLayerNumber - 1][neuron] * (1.0 - h[middleLayerNumber - 1][neuron]) * sumDelta;

        // 学習
//        std::cout << "[learn] before m: " << middleNeurons[middleLayerNumber - 1][neuron].toString() << std::endl;
        middleNeurons[middleLayerNumber - 1][neuron].learn(delta, h[middleLayerNumber - 2]);
//        std::cout << "[learn] after m: " << middleNeurons[middleLayerNumber - 1][neuron].toString() << std::endl;
    }
}

void MultiLayerPerceptron::middleMiddleLayerLearnThread(const std::vector<std::vector<double>> h, const int begin,
                                                        const int end) {
    for (int neuron = begin; neuron < end; ++neuron) {
        for (int layer = (int)middleLayerNumber - 2; layer >= 1; --layer) {
            // 中間層ニューロンの学習係数deltaを計算
            double sumDelta = 0.0;
            for (int k = 0; k < middleNumber; ++k) {
                Neuron n = middleNeurons[layer + 1][k];
                sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
            }
            double delta = h[layer][neuron] * (1.0 - h[layer][neuron]) * sumDelta;

            // 学習
//                std::cout << "[learn] before m: " << middleNeurons[layer][neuron].toString() << std::endl;
            middleNeurons[layer][neuron].learn(delta, h[layer - 1]);
//                std::cout << "[learn] after m: " << middleNeurons[layer][neuron].toString() << std::endl;
        }
    }
}

void MultiLayerPerceptron::middleFirstLayerLearnThread(const std::vector<std::vector<double>> h,
                                                       const std::vector<double> in, const int begin, const int end) {
    for (int neuron = begin; neuron < end; ++neuron) {
        // 中間層ニューロンの学習係数deltaを計算
        double sumDelta = 0.0;

        if (middleLayerNumber > 1) {
            for (int k = 0; k < middleNumber; ++k) {
                Neuron n = middleNeurons[1][k];
                sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
            }
        } else {
            for (int k = 0; k < outputNumber; ++k) {
                Neuron n = outputNeurons[k];
                sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
            }
        }
        double delta = h[0][neuron] * (1.0 - h[0][neuron]) * sumDelta;

        // 学習
//            std::cout << "[learn] before m: " << middleNeurons[0][neuron].toString() << std::endl;
        middleNeurons[0][neuron].learn(delta, in);
//            std::cout << "[learn] after m: " << middleNeurons[0][neuron].toString() << std::endl;
    }
}


void MultiLayerPerceptron::out(std::vector<double> input){
    std::vector<std::vector<double>> h = std::vector<std::vector<double>>(middleLayerNumber, std::vector<double>(middleNumber, 0));
    std::vector<double> o = std::vector<double>(outputNumber, 0);

    for (int neuron = 0; neuron < middleNumber; ++neuron) {
        h[0][neuron] = middleNeurons[0][neuron].output(input);
    }

    for (int layer = 1; layer < middleLayerNumber; ++layer) {
        for (int neuron = 0; neuron < middleNumber; ++neuron) {
            h[layer][neuron] = middleNeurons[layer][neuron].output(h[layer - 1]);
        }
    }

    for (int neuron = 0; neuron < outputNumber; ++neuron) {
        o[neuron] = outputNeurons[neuron].output(h[middleLayerNumber - 1]);
    }

    for (int neuron = 0; neuron < outputNumber; ++neuron) {
        std::cout << "output[" << neuron << "]: " << o[neuron] << " ";
    }
    std::cout << std::endl;
}