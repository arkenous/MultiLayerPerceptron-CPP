//
// Created by Kensuke Kosaka on 2016/10/27.
//

#include "MultiLayerPerceptron.h"
#include "Neuron.h"
#include "iostream"
#include <thread>
#include <sstream>

using namespace std;

/**
 * MultiLayerPerceptronのコンストラクタ
 * @param input 入力層のニューロン数
 * @param middle 中間層のニューロン数
 * @param output 出力層のニューロン数
 * @param middleLayer 中間層の層数
 * @param middleLayerType 中間層の活性化関数の種類指定．0: identity 1: sigmoid 2: tanh 3: ReLU
 * @param dropout_rate Dropout率
 * @param sda_params Stacked Denoising Autoencoderのパラメータ
 * @return
 */
MultiLayerPerceptron::MultiLayerPerceptron(unsigned long input, unsigned long middle, unsigned long output,
                                           unsigned long middleLayer, int middleLayerType, double dropout_rate,
                                           std::string sda_params) {
  sda_neurons = setupSdA(sda_params);

  this->inputNumber = sda_neurons[sda_neurons.size() - 1].size(); // SdAの末尾レイヤの出力数がMLPの入力数となる
  this->middleNumber = middle;
  this->outputNumber = output;
  this->middleLayerNumber = middleLayer;
  this->middleLayerType = middleLayerType;

  this->middleNeurons.resize(middleLayerNumber);
  for (std::vector<Neuron> n : this->middleNeurons) n.resize(middleNumber);

  std::vector<Neuron> neuronPerLayer(middleNumber);

  std::vector<double> emptyVector;

  for (int layer = 0; layer < middleLayerNumber; ++layer) {
    if (layer == 0) {
      for (int neuron = 0; neuron < middleNumber; ++neuron) {
        // 中間層の最初の層については，SdA末尾レイヤのニューロン数がニューロンへの入力数となる
        neuronPerLayer[neuron] = Neuron(inputNumber, emptyVector, emptyVector, emptyVector, emptyVector, emptyVector,
                                        0, 0.0, middleLayerType, dropout_rate);
      }
    } else {
      for (int neuron = 0; neuron < middleNumber; ++neuron) {
        // それ以降の層については，中間層の各層のニューロン数がニューロンへの入力数となる
        neuronPerLayer[neuron] = Neuron(middleNumber, emptyVector, emptyVector, emptyVector, emptyVector, emptyVector,
                                        0, 0.0, middleLayerType, dropout_rate);
      }
    }
    this->middleNeurons[layer] = neuronPerLayer;
  }

  this->outputNeurons.resize(outputNumber);
  for (int neuron = 0; neuron < output; ++neuron) {
    this->outputNeurons[neuron] = Neuron(middleNumber, emptyVector, emptyVector, emptyVector, emptyVector, emptyVector,
                                         0, 0.0, 1, dropout_rate);
  }
}

std::vector<std::vector<Neuron>> MultiLayerPerceptron::setupSdA(std::string sda_params) {
  std::stringstream ss(sda_params);
  std::string item;
  std::vector<std::string> elemsPerSdA;
  std::vector<std::string> elemsPerNeuron;
  std::vector<std::string> elemsPerParam;

  // $ でSdA単位で分割する（= SdA層のレイヤ数）
  while (std::getline(ss, item, '$')) if (!item.empty()) elemsPerSdA.push_back(item);
  sda_neurons.resize(elemsPerSdA.size());
  sda_out.resize(elemsPerSdA.size());
  item = "";
  ss.str("");
  ss.clear(stringstream::goodbit);

  for (int sda = 0; sda < elemsPerSdA.size(); ++sda) {
    // ' でニューロン単位で分割する
    ss = std::stringstream(elemsPerSdA[sda]);
    while (std::getline(ss, item, '\'')) if (!item.empty()) elemsPerNeuron.push_back(item);
    sda_neurons[sda].resize(elemsPerNeuron.size());
    sda_out[sda].resize(elemsPerNeuron.size());
    item = "";
    ss.str("");
    ss.clear(stringstream::goodbit);

    for (int neuron = 0; neuron < elemsPerNeuron.size(); ++neuron) {
      // パラメータごとに分割する
      ss = std::stringstream(elemsPerNeuron[neuron]);
      while (std::getline(ss, item, '|')) if (!item.empty()) elemsPerParam.push_back(item);
      item = "";
      ss.str("");
      ss.clear(stringstream::goodbit);

      double bias = std::stod(elemsPerParam.back());
      elemsPerParam.pop_back();

      int iteration = std::stoi(elemsPerParam.back());
      elemsPerParam.pop_back();

      std::vector<double> weight = separate_by_camma(elemsPerParam[0]);
      std::vector<double> m = separate_by_camma(elemsPerParam[1]);
      std::vector<double> nu = separate_by_camma(elemsPerParam[2]);
      std::vector<double> m_hat = separate_by_camma(elemsPerParam[3]);
      std::vector<double> nu_hat = separate_by_camma(elemsPerParam[4]);

      sda_neurons[sda][neuron] = Neuron(weight.size(), weight, m, nu, m_hat, nu_hat, iteration, bias, 1, 0.0);

      elemsPerParam.clear();
    }
    elemsPerNeuron.clear();
  }
  elemsPerSdA.clear();

  return sda_neurons;
}

std::vector<double> MultiLayerPerceptron::separate_by_camma(std::string input) {
  std::vector<double> result;
  std::stringstream ss = std::stringstream(input);
  std::string item;
  while (std::getline(ss, item, ',')) if (!item.empty()) result.push_back(std::stod(item));
  item = "";
  ss.str("");
  ss.clear(stringstream::goodbit);

  return result;
}

/**
 * 教師入力データと教師出力データを元にニューラルネットワークを学習する
 * @param x 二次元の教師入力データ，データセット * データ
 * @param answer 教師入力データに対応した二次元の教師出力データ，データセット * データ
 */
void MultiLayerPerceptron::learn(std::vector<std::vector<double>> x, std::vector<std::vector<double>> answer) {
  h = std::vector<std::vector<double>>(middleLayerNumber, std::vector<double>(middleNumber, 0.0));
  o = std::vector<double>(outputNumber, 0.0);

  int succeed = 0; // 連続正解回数のカウンタを初期化

  std::random_device rnd; // 非決定的乱数生成器
  std::mt19937 mt; // メルセンヌ・ツイスタ
  mt.seed(rnd());
  std::uniform_real_distribution<double> real_rnd(0.0, 1.0); // 0.0以上1.0未満の範囲で値を生成する

  for (int trial = 0; trial < this->MAX_TRIAL; ++trial) {
    for (int layer = 0; layer < middleLayerNumber; ++layer) {
      for (int neuron = 0; neuron < middleNumber; ++neuron) {
        middleNeurons[layer][neuron].dropout(real_rnd(mt));
      }
    }
    for (int neuron = 0; neuron < outputNumber; ++neuron) {
      outputNeurons[neuron].dropout(1.0); // 出力層ニューロンはDropoutさせない
    }

    // 使用する教師データを選択
    std::vector<double> in = x[trial % answer.size()]; // 利用する教師入力データ
    std::vector<double> ans = answer[trial % answer.size()]; // 教師出力データ

    // Feed Forward
    // SdA First Layer
    std::vector<std::thread> threads(num_thread);
    int charge = 1;
    threads.clear();
    if (sda_neurons[0].size() <= num_thread) charge = 1;
    else charge = sda_neurons[0].size() / num_thread;
    for (int i = 0; i < sda_neurons[0].size(); i += charge) {
      if (i != 0 && sda_neurons[0].size() / i == 1) {
        threads.push_back(std::thread(&MultiLayerPerceptron::sdaFirstLayerOutThread, this,
                                      std::ref(in), i, sda_neurons[0].size()));
      } else {
        threads.push_back(std::thread(&MultiLayerPerceptron::sdaFirstLayerOutThread, this,
                                      std::ref(in), i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();

    // SdA Other Layer
    for (int layer = 1; layer <= (int) sda_neurons.size() - 1; ++layer) {
      threads.clear();
      if (sda_neurons[layer].size() <= num_thread) charge = 1;
      else charge = sda_neurons[layer].size() / num_thread;
      for (int i = 0; i < sda_neurons[layer].size(); i += charge) {
        if (i != 0 && sda_neurons[layer].size() / i == 1) {
          threads.push_back(std::thread(&MultiLayerPerceptron::sdaOtherLayerOutThread, this,
                                        layer, i, sda_neurons[layer].size()));
        } else {
          threads.push_back(std::thread(&MultiLayerPerceptron::sdaOtherLayerOutThread, this,
                                        layer, i, i + charge));
        }
      }
      for (std::thread &th : threads) th.join();
    }


    // 1層目の中間層の出力計算
    threads.clear();
    if (middleNumber <= num_thread) charge = 1;
    else charge = middleNumber / num_thread;
    for (int i = 0; i < middleNumber; i += charge) {
      if (i != 0 && middleNumber / i == 1) {
        threads.push_back(std::thread(&MultiLayerPerceptron::middleFirstLayerForwardThread, this, i, middleNumber));
      } else {
        threads.push_back(std::thread(&MultiLayerPerceptron::middleFirstLayerForwardThread, this, i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();

    // 一つ前の中間層より得られた出力を用いて，以降の中間層を順に計算
    if (middleNumber <= num_thread) charge = 1;
    else charge = middleNumber / num_thread;
    for (int layer = 1; layer <= (int) middleLayerNumber - 1; ++layer) {
      threads.clear();
      for (int i = 0; i < middleNumber; i += charge) {
        if (i != 0 && middleNumber / i == 1) {
          threads.push_back(std::thread(&MultiLayerPerceptron::middleLayerForwardThread, this, layer, i, middleNumber));
        } else {
          threads.push_back(std::thread(&MultiLayerPerceptron::middleLayerForwardThread, this, layer, i, i + charge));
        }
      }
      for (std::thread &th : threads) th.join();
    }

    // 出力値を推定：中間層の最終層の出力を用いて，出力層の出力計算
    threads.clear();
    if (outputNumber <= num_thread) charge = 1;
    else charge = outputNumber / num_thread;
    for (int i = 0; i < outputNumber; i += charge) {
      if (i != 0 && outputNumber / i == 1) {
        threads.push_back(std::thread(&MultiLayerPerceptron::outForwardThread, this, i, outputNumber));
      } else {
        threads.push_back(std::thread(&MultiLayerPerceptron::outForwardThread, this, i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();

    successFlg = true;

    // Back Propagation (learn phase)
    //region 出力層を学習する
    threads.clear();
    if (outputNumber <= num_thread) charge = 1;
    else charge = outputNumber / num_thread;
    for (int i = 0; i < outputNumber; i += charge) {
      if (i != 0 && outputNumber / i == 1) {
        threads.push_back(
            std::thread(&MultiLayerPerceptron::outLearnThread, this, std::ref(in), std::ref(ans), i, outputNumber));
      } else {
        threads.push_back(
            std::thread(&MultiLayerPerceptron::outLearnThread, this, std::ref(in), std::ref(ans), i, i + charge));
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
      if (middleNumber <= num_thread) charge = 1;
      else charge = middleNumber / num_thread;
      for (int i = 0; i < middleNumber; i += charge) {
        if (i != 0 && middleNumber / i == 1) {
          threads.push_back(std::thread(&MultiLayerPerceptron::middleLastLayerLearnThread, this, i, middleNumber));
        } else {
          threads.push_back(std::thread(&MultiLayerPerceptron::middleLastLayerLearnThread, this, i, i + charge));
        }
      }
      for (std::thread &th : threads) th.join();
    }
    //endregion

    //region 出力層と入力層に最も近い層一つずつを除いた残りの中間層を入力層に向けて学習する
    if (middleNumber <= num_thread) charge = 1;
    else charge = middleNumber / num_thread;
    for (int layer = (int) middleLayerNumber - 2; layer >= 1; --layer) {
      threads.clear();
      for (int i = 0; i < middleNumber; i += charge) {
        if (i != 0 && middleNumber / i == 1) {
          threads.push_back(
              std::thread(&MultiLayerPerceptron::middleMiddleLayerLearnThread, this, layer, i, middleNumber));
        } else {
          threads.push_back(
              std::thread(&MultiLayerPerceptron::middleMiddleLayerLearnThread, this, layer, i, i + charge));
        }
      }
      for (std::thread &th : threads) th.join();
    }
    //endregion

    //region 中間層の最初の層を学習する
    threads.clear();
    if (middleNumber <= num_thread) charge = 1;
    else charge = middleNumber / num_thread;
    for (int i = 0; i < middleNumber; i += charge) {
      if (i != 0 && middleNumber / i == 1) {
        threads.push_back(std::thread(&MultiLayerPerceptron::middleFirstLayerLearnThread, this, i, middleNumber));
      } else {
        threads.push_back(std::thread(&MultiLayerPerceptron::middleFirstLayerLearnThread, this, i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();
    //endregion

    if (sda_neurons.size() > 1) {
      threads.clear();
      if (sda_neurons[sda_neurons.size() - 1].size() <= num_thread) charge = 1;
      else charge = sda_neurons[sda_neurons.size() - 1].size() / num_thread;
      for (int i = 0; i < sda_neurons[sda_neurons.size() - 1].size(); i += charge) {
        if (i != 0 && sda_neurons[sda_neurons.size() - 1].size() / i == 1) {
          threads.push_back(std::thread(&MultiLayerPerceptron::sdaLastLayerLearnThread, this,
                                        i, sda_neurons[sda_neurons.size() - 1].size()));
        } else {
          threads.push_back(std::thread(&MultiLayerPerceptron::sdaLastLayerLearnThread, this,
                                        i, i + charge));
        }
      }
      for (std::thread &th : threads) th.join();
    }

    for (int layer = sda_neurons.size() - 2; layer >= 1; --layer) {
      if (sda_neurons[layer].size() <= num_thread) charge = 1;
      else charge = sda_neurons[layer].size() / num_thread;
      threads.clear();
      for (int i = 0; i < sda_neurons[layer].size(); i += charge) {
        if (i != 0 && sda_neurons[layer].size() / i == 1) {
          threads.push_back(std::thread(&MultiLayerPerceptron::sdaMiddleLayerLearnThread, this,
                                        layer, i, sda_neurons[layer].size()));
        } else {
          threads.push_back(std::thread(&MultiLayerPerceptron::sdaMiddleLayerLearnThread, this,
                                        layer, i, i + charge));
        }
      }
      for (std::thread &th : threads) th.join();
    }

    threads.clear();
    if (sda_neurons[0].size() <= num_thread) charge = 1;
    else charge = sda_neurons[0].size() / num_thread;
    for (int i = 0; i < sda_neurons[0].size(); i += charge) {
      if (i != 0 && sda_neurons[0].size() / i == 1) {
        threads.push_back(std::thread(&MultiLayerPerceptron::sdaFirstLayerLearnThread, this,
                                      std::ref(in), i, sda_neurons[0].size()));
      } else {
        threads.push_back(std::thread(&MultiLayerPerceptron::sdaFirstLayerLearnThread, this,
                                      std::ref(in), i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();

    //endregion
  }

  // 全ての教師データで正解を出すか，収束限度回数を超えた場合に終了
}

/**
 * ニューラルネットワークの状態をまとめた文字列を返す
 * @return  ニューラルネットワークの状態（重み付け）をまとめた文字列
 */
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

void MultiLayerPerceptron::sdaFirstLayerOutThread(const std::vector<double> in, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) sda_out[0][neuron] = sda_neurons[0][neuron].output(in);
}

void MultiLayerPerceptron::sdaOtherLayerOutThread(const int layer, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    sda_out[layer][neuron] = sda_neurons[layer][neuron].output(sda_out[layer - 1]);
  }
}

void MultiLayerPerceptron::middleFirstLayerForwardThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // SdAの最終層の出力を入れる
    h[0][neuron] = middleNeurons[0][neuron].learn_output(sda_out[sda_out.size() - 1]);
  }
}

void MultiLayerPerceptron::middleLayerForwardThread(const int layer, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    h[layer][neuron] = middleNeurons[layer][neuron].learn_output(h[layer - 1]);
  }
}

void MultiLayerPerceptron::outForwardThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    o[neuron] = outputNeurons[neuron].learn_output(h[middleLayerNumber - 1]);
  }
}

/**
 * 出力層の学習，スレッドを用いて並列学習するため，学習するニューロンの開始点と終了点も必要
 * 誤差関数には交差エントロピーを，活性化関数にシグモイド関数を用いるため，deltaは 出力 - 教師出力 で得られる
 * @param in 入力データ
 * @param ans 教師出力データ
 * @param o 出力層の出力データ
 * @param h 中間層の出力データ
 * @param begin 学習するニューロンセットの開始点
 * @param end 学習するニューロンセットの終了点
 */
void MultiLayerPerceptron::outLearnThread(const std::vector<double> in, const std::vector<double> ans,
                                          const int begin, const int end) {
  // Dropoutを用いた学習済みNNの出力を得るようにする
  std::vector<double> output = this->out(in, false);
  for (int neuron = begin; neuron < end; ++neuron) {
    // 出力層ニューロンのdeltaの計算
    double delta = o[neuron] - ans[neuron];

    // 教師データとの誤差が十分小さい場合は学習しない．そうでなければ正解フラグをfalseに
    if (std::abs(ans[neuron] - output[neuron]) < MAX_GAP) continue;
    else successFlg = false;

    // 出力層の学習
    outputNeurons[neuron].learn(delta, h[middleLayerNumber - 1]);
  }
}

/**
 * 中間層の最終層の学習．中間層の層数が2以上の場合のみこれを使う．
 * 活性化関数に何を使うかで，deltaの計算式が変わる
 * @param h 中間層の出力データ
 * @param begin 学習するニューロンセットの開始点
 * @param end 学習するニューロンセットの終了点
 */
void MultiLayerPerceptron::middleLastLayerLearnThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // 中間層ニューロンのdeltaを計算
    double sumDelta = 0.0;
    for (int k = 0; k < outputNumber; ++k) {
      Neuron n = outputNeurons[k];
      sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
    }

    // どの活性化関数を用いるかで，deltaの計算方法が変わる
    double delta;
    if (middleLayerType == 0) {
      delta = 1.0 * sumDelta;
    } else if (middleLayerType == 1) {
      delta = (h[middleLayerNumber - 1][neuron] * (1.0 - h[middleLayerNumber - 1][neuron])) * sumDelta;
    } else if (middleLayerType == 2) {
      delta = (1.0 - pow(h[middleLayerNumber - 1][neuron], 2)) * sumDelta;
    } else {
      // ReLU
      if (h[middleLayerNumber - 1][neuron] > 0) delta = 1.0 * sumDelta;
      else delta = 0 * sumDelta;
    }

    // 学習
    middleNeurons[middleLayerNumber - 1][neuron].learn(delta, h[middleLayerNumber - 2]);
  }
}

/**
 * 出力層と入力層に最も近い層一つずつを除いた残りの中間層を入力層に向けて学習する．中間層が3層以上の場合にこれを使う．
 * @param h 中間層の出力データ
 * @param begin 学習するニューロンセットの開始点
 * @param end 学習するニューロンセットの終了点
 */
void MultiLayerPerceptron::middleMiddleLayerLearnThread(const int layer, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // 中間層ニューロンのdeltaを計算
    double sumDelta = 0.0;
    for (int k = 0; k < middleNumber; ++k) {
      Neuron n = middleNeurons[layer + 1][k];
      sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
    }

    double delta;
    if (middleLayerType == 0) {
      delta = 1.0 * sumDelta;
    } else if (middleLayerType == 1) {
      delta = (h[layer][neuron] * (1.0 - h[layer][neuron])) * sumDelta;
    } else if (middleLayerType == 2) {
      delta = (1.0 - pow(h[layer][neuron], 2)) * sumDelta;
    } else {
      // ReLU
      if (h[layer][neuron] > 0) delta = 1.0 * sumDelta;
      else delta = 0 * sumDelta;
    }

    // 学習
    middleNeurons[layer][neuron].learn(delta, h[layer - 1]);
  }
}

/**
 * 中間層の最初の層を学習する
 * @param h 中間層の出力データ
 * @param in 教師入力データ
 * @param begin 学習するニューロンセットの開始点
 * @param end 学習するニューロンセットの終了点
 */
void MultiLayerPerceptron::middleFirstLayerLearnThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // 中間層ニューロンのdeltaを計算
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

    double delta;
    if (middleLayerType == 0) {
      delta = 1.0 * sumDelta;
    } else if (middleLayerType == 1) {
      delta = (h[0][neuron] * (1.0 - h[0][neuron])) * sumDelta;
    } else if (middleLayerType == 2) {
      delta = (1.0 - pow(h[0][neuron], 2)) * sumDelta;
    } else {
      // ReLU
      if (h[0][neuron] > 0) delta = 1.0 * sumDelta;
      else delta = 0 * sumDelta;
    }

    // 学習
    middleNeurons[0][neuron].learn(delta, sda_out[sda_out.size() - 1]);
  }
}

void MultiLayerPerceptron::sdaLastLayerLearnThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    double sumDelta = 0.0;
    for (int k = 0; k < middleNeurons[0].size(); ++k) {
      Neuron n = middleNeurons[0][k];
      sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
    }

    double delta;
    // sigmoid
    delta = (sda_out[sda_out.size() - 1][neuron] * (1.0 - sda_out[sda_out.size() - 1][neuron])) * sumDelta;

    sda_neurons[sda_neurons.size() - 1][neuron].learn(delta, sda_out[sda_out.size() - 2]);
  }
}

void MultiLayerPerceptron::sdaMiddleLayerLearnThread(const int layer, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    double sumDelta = 0.0;
    for (int k = 0; k < sda_neurons[layer + 1].size(); ++k) {
      Neuron n = sda_neurons[layer + 1][k];
      sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
    }

    double delta;
    // sigmoid
    delta = (sda_out[layer][neuron] * (1.0 - sda_out[layer][neuron])) * sumDelta;

    sda_neurons[layer][neuron].learn(delta, sda_out[layer - 1]);
  }
}

void MultiLayerPerceptron::sdaFirstLayerLearnThread(const std::vector<double> in, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    double sumDelta = 0.0;

    if (sda_neurons.size() > 1) {
      for (int k = 0; k < sda_neurons[1].size(); ++k) {
        Neuron n = sda_neurons[1][k];
        sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
      }
    } else {
      for (int k = 0; k < middleNeurons[0].size(); ++k) {
        Neuron n = middleNeurons[0][k];
        sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
      }
    }

    double delta;
    // sigmoid
    delta = (sda_out[0][neuron] * (1.0 - sda_out[0][neuron])) * sumDelta;

    sda_neurons[0][neuron].learn(delta, in);
  }
}

/**
 * 与えられたデータをニューラルネットワークに入力し，出力を返す
 * @param input ニューラルネットワークに入力するデータ
 * @param showResult 結果をコンソールに出力するかを指定する
 */
std::vector<double> MultiLayerPerceptron::out(std::vector<double> input, bool showResult) {
  // Feed Forward
  // SdA First Layer
  std::vector<std::thread> threads(num_thread);
  int charge = 1;
  threads.clear();
  if (sda_neurons[0].size() <= num_thread) charge = 1;
  else charge = sda_neurons[0].size() / num_thread;
  for (int i = 0; i < sda_neurons[0].size(); i += charge) {
    if (i != 0 && sda_neurons[0].size() / i == 1) {
      threads.push_back(std::thread(&MultiLayerPerceptron::sdaFirstLayerOutThread, this,
                                    std::ref(input), i, sda_neurons[0].size()));
    } else {
      threads.push_back(std::thread(&MultiLayerPerceptron::sdaFirstLayerOutThread, this,
                                    std::ref(input), i, i + charge));
    }
  }
  for (std::thread &th : threads) th.join();

  // SdA Other Layer
  for (int layer = 1; layer <= (int) sda_neurons.size() - 1; ++layer) {
    threads.clear();
    if (sda_neurons[layer].size() <= num_thread) charge = 1;
    else charge = sda_neurons[layer].size() / num_thread;
    for (int i = 0; i < sda_neurons[layer].size(); i += charge) {
      if (i != 0 && sda_neurons[layer].size() / i == 1) {
        threads.push_back(std::thread(&MultiLayerPerceptron::sdaOtherLayerOutThread, this,
                                      layer, i, sda_neurons[layer].size()));
      } else {
        threads.push_back(std::thread(&MultiLayerPerceptron::sdaOtherLayerOutThread, this,
                                      layer, i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();
  }


  learnedH = std::vector<std::vector<double>>(middleLayerNumber, std::vector<double>(middleNumber, 0));
  learnedO = std::vector<double>(outputNumber, 0);

  threads.clear();
  if (middleNumber <= num_thread) charge = 1;
  else charge = middleNumber / num_thread;
  for (int i = 0; i < middleNumber; i += charge) {
    if (i != 0 && middleNumber / i == 1) {
      threads.push_back(std::thread(&MultiLayerPerceptron::middleFirstLayerOutThread, this, i, middleNumber));
    } else {
      threads.push_back(std::thread(&MultiLayerPerceptron::middleFirstLayerOutThread, this, i, i + charge));
    }
  }
  for (std::thread &th : threads) th.join();

  if (middleNumber <= num_thread) charge = 1;
  else charge = middleNumber / num_thread;
  for (int layer = 1; layer <= (int) middleLayerNumber - 1; ++layer) {
    threads.clear();
    for (int i = 0; i < middleNumber; i += charge) {
      if (i != 0 && middleNumber / i == 1) {
        threads.push_back(std::thread(&MultiLayerPerceptron::middleLayerOutThread, this, layer, i, middleNumber));
      } else {
        threads.push_back(std::thread(&MultiLayerPerceptron::middleLayerOutThread, this, layer, i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();
  }

  threads.clear();
  if (outputNumber <= num_thread) charge = 1;
  else charge = outputNumber / num_thread;
  for (int i = 0; i < outputNumber; i += charge) {
    if (i != 0 && outputNumber / i == 1) {
      threads.push_back(std::thread(&MultiLayerPerceptron::outOutThread, this, i, outputNumber));
    } else {
      threads.push_back(std::thread(&MultiLayerPerceptron::outOutThread, this, i, i + charge));
    }
  }
  for (std::thread &th : threads) th.join();

  if (showResult) {
    for (int neuron = 0; neuron < outputNumber; ++neuron) {
      std::cout << "output[" << neuron << "]: " << learnedO[neuron] << " ";
    }
    std::cout << std::endl;
  }

  return learnedO;
}

void MultiLayerPerceptron::middleFirstLayerOutThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    learnedH[0][neuron] = middleNeurons[0][neuron].output(sda_out[sda_out.size() - 1]);
  }
}

void MultiLayerPerceptron::middleLayerOutThread(const int layer, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    learnedH[layer][neuron] = middleNeurons[layer][neuron].output(learnedH[layer - 1]);
  }
}

void MultiLayerPerceptron::outOutThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    learnedO[neuron] = outputNeurons[neuron].output(learnedH[middleLayerNumber - 1]);
  }
}
