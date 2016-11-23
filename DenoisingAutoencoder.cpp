
#include <iostream>
#include <sstream>
#include "DenoisingAutoencoder.h"
#include <thread>

using namespace std;

DenoisingAutoencoder::DenoisingAutoencoder(unsigned long num_input, float compression_rate) {
  input_neuron_num = num_input;
  middle_neuron_num = (unsigned long) (num_input * (1.0 - compression_rate));
  output_neuron_num = num_input;
  middle_layer_type = 1;

  std::vector<double> emptyVector;
  middle_neurons.resize(middle_neuron_num);
  for (int neuron = 0; neuron < middle_neuron_num; ++neuron) {
    middle_neurons[neuron] = Neuron(input_neuron_num, emptyVector, emptyVector, emptyVector, emptyVector, emptyVector,
                                    0, 0.0, middle_layer_type, 0.0);
  }

  output_neurons.resize(output_neuron_num);
  for (int neuron = 0; neuron < output_neuron_num; ++neuron) {
    output_neurons[neuron] = Neuron(middle_neuron_num, emptyVector, emptyVector, emptyVector, emptyVector, emptyVector,
                                    0, 0.0, 0, 0.0);
  }

  h.resize(middle_neuron_num);
  o.resize(output_neuron_num);
  learnedH.resize(middle_neuron_num);
  learnedO.resize(output_neuron_num);
}

std::string DenoisingAutoencoder::learn(std::vector<std::vector<double>> input,
                                        std::vector<std::vector<double>> noisy_input) {
  int succeed = 0; // 連続正解回数のカウンタを初期化

  for (int trial = 0; trial < MAX_TRIAL; ++trial) {
    // Dropoutは無効にする
    for (int neuron = 0; neuron < middle_neuron_num; ++neuron) middle_neurons[neuron].dropout(1.0);
    for (int neuron = 0; neuron < output_neuron_num; ++neuron) output_neurons[neuron].dropout(1.0);

    // 使用する教師データを選択
    std::vector<double> in = noisy_input[trial % input.size()];
    std::vector<double> ans = input[trial % input.size()];

    // Feed Forward
    std::vector<std::thread> threads(num_thread);
    unsigned long charge = 1;
    threads.clear();
    if (middle_neuron_num <= num_thread) { charge = 1; }
    else { charge = middle_neuron_num / num_thread; }
    for (int i = 0; i < middle_neuron_num; i += charge) {
      if (i != 0 && middle_neuron_num / i == 1) {
        threads.push_back(std::thread(&DenoisingAutoencoder::middleForwardThread, this,
                                      std::ref(in), i, middle_neuron_num));
      } else {
        threads.push_back(std::thread(&DenoisingAutoencoder::middleForwardThread, this,
                                      std::ref(in), i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();

    threads.clear();
    if (output_neuron_num <= num_thread) { charge = 1; }
    else { charge = output_neuron_num / num_thread; }
    for (int i = 0; i < output_neuron_num; i += charge) {
      if (i != 0 && output_neuron_num / i == 1) {
        threads.push_back(std::thread(&DenoisingAutoencoder::outForwardThread, this, i, output_neuron_num));
      } else {
        threads.push_back(std::thread(&DenoisingAutoencoder::outForwardThread, this, i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();

    successFlg = true;

    // Back Propagation (learn phase)
    threads.clear();
    if (output_neuron_num <= num_thread) { charge = 1; }
    else { charge = output_neuron_num / num_thread; }
    for (int i = 0; i < output_neuron_num; i += charge) {
      if (i != 0 && output_neuron_num / i == 1) {
        threads.push_back(std::thread(&DenoisingAutoencoder::outLearnThread, this,
                                      std::ref(in), std::ref(ans), i, output_neuron_num));
      } else {
        threads.push_back(std::thread(&DenoisingAutoencoder::outLearnThread, this,
                                      std::ref(in), std::ref(ans), i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();

    if (successFlg) {
      succeed++;
      if (succeed >= input.size()) { break; }
      else { continue; }
    } else { succeed = 0; }

    threads.clear();
    if (middle_neuron_num <= num_thread) { charge = 1; }
    else { charge = middle_neuron_num / num_thread; }
    for (int i = 0; i < middle_neuron_num; i += charge) {
      if (i != 0 && middle_neuron_num / i == 1) {
        threads.push_back(std::thread(&DenoisingAutoencoder::middleLearnThread, this,
                                      std::ref(in), i, middle_neuron_num));
      } else {
        threads.push_back(std::thread(&DenoisingAutoencoder::middleLearnThread, this,
                                      std::ref(in), i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();
  }

  // 全ての教師データで正解を出すか，収束限度回数を超えた場合に終了
  // エンコーダ部分である中間層ニューロンの各パラメータをstd::stringに詰める
  std::stringstream ss;
  for (int neuron = 0; neuron < middle_neuron_num; ++neuron) {
    // 重みを詰める
    for (int weight_num = 0; weight_num < input_neuron_num; ++weight_num) {
      ss << middle_neurons[neuron].getInputWeightIndexOf(weight_num) << ',';
    }
    ss << '|';

    // Adamのmを詰める
    for (int mNum = 0; mNum < input_neuron_num; ++mNum) {
      ss << middle_neurons[neuron].getMIndexOf(mNum) << ',';
    }
    ss << '|';

    // Adamのnuを詰める
    for (int nuNum = 0; nuNum < input_neuron_num; ++nuNum) {
      ss << middle_neurons[neuron].getNuIndexOf(nuNum) << ',';
    }
    ss << '|';

    // Adamのm_hatを詰める
    for (int mHatNum = 0; mHatNum < input_neuron_num; ++mHatNum) {
      ss << middle_neurons[neuron].getMHatIndexOf(mHatNum) << ',';
    }
    ss << '|';

    // Adamのnu_hatを詰める
    for (int nuHatNum = 0; nuHatNum < input_neuron_num; ++nuHatNum) {
      ss << middle_neurons[neuron].getNuHatIndexOf(nuHatNum) << ',';
    }
    ss << '|';

    // Adamのiterationを詰める
    ss << middle_neurons[neuron].getIteration() << '|';

    // バイアスを入れ，最後に ' を入れる
    ss << middle_neurons[neuron].getBias() << '\'';
  }

  std::string neuron_params = ss.str();
  // 末尾の ' を削除する
  neuron_params.pop_back();
  ss.str("");
  ss.clear(stringstream::goodbit);

  return neuron_params;
}

std::vector<std::vector<double>> DenoisingAutoencoder::getMiddleOutput(std::vector<std::vector<double>> noisy_input) {
  std::vector<std::vector<double>> middle_output(noisy_input.size());
  std::vector<std::thread> threads(num_thread);
  unsigned long charge = 1;

  for (int set = 0; set < noisy_input.size(); ++set) {
    threads.clear();
    if (middle_neuron_num <= num_thread) { charge = 1; }
    else { charge = middle_neuron_num / num_thread; }
    for (int i = 0; i < middle_neuron_num; i += charge) {
      if (i != 0 && middle_neuron_num / i == 1) {
        threads.push_back(std::thread(&DenoisingAutoencoder::middleOutThread, this,
                                      std::ref(noisy_input[set]), i, middle_neuron_num));
      } else {
        threads.push_back(std::thread(&DenoisingAutoencoder::middleOutThread, this,
                                      std::ref(noisy_input[set]), i, i + charge));
      }
    }
    for (std::thread &th : threads) th.join();
    middle_output[set] = learnedH;
  }

  return middle_output;
}

void DenoisingAutoencoder::middleForwardThread(const std::vector<double> in, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) h[neuron] = middle_neurons[neuron].learn_output(in);
}

void DenoisingAutoencoder::outForwardThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) o[neuron] = output_neurons[neuron].learn_output(h);
}

void DenoisingAutoencoder::outLearnThread(const std::vector<double> in, const std::vector<double> ans,
                                          const int begin, const int end) {
  // Dropoutを用いた学習済みNNの出力を得るようにする
  std::vector<double> output = out(in, false);
  for (int neuron = begin; neuron < end; ++neuron) {
    // 出力層ニューロンのdeltaの計算
    double delta = o[neuron] - ans[neuron];

    // 教師データとの誤差が十分小さい場合は学習しない．そうでなければ正解フラグをfalseに
    if (mean_squared_error(o[neuron], ans[neuron]) < MAX_GAP) { continue; }
    else { successFlg = false; }

    // 出力層の学習
    output_neurons[neuron].learn(delta, h);
  }
}

void DenoisingAutoencoder::middleLearnThread(const std::vector<double> in, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // 中間層ニューロンのdeltaを計算
    double sumDelta = 0.0;

    for (int k = 0; k < output_neuron_num; ++k) {
      Neuron n = output_neurons[k];
      sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
    }

    double delta;
    if (middle_layer_type == 0) { delta = 1.0 * sumDelta; }
    else if (middle_layer_type == 1) { delta = (h[neuron] * (1.0 - h[neuron])) * sumDelta; }
    else if (middle_layer_type == 2) { delta = (1.0 - pow(h[neuron], 2)) * sumDelta; }
    else {
      //ReLU
      if (h[neuron] > 0) { delta = 1.0 * sumDelta; }
      else { delta = 0 * sumDelta; }
    }

    // 学習
    middle_neurons[neuron].learn(delta, in);
  }
}

std::vector<double> DenoisingAutoencoder::out(std::vector<double> input, bool showResult) {
  std::vector<std::thread> threads(num_thread);
  unsigned long charge = 1;
  threads.clear();
  if (middle_neuron_num <= num_thread) { charge = 1; }
  else { charge = middle_neuron_num / num_thread; }
  for (int i = 0; i < middle_neuron_num; i += charge) {
    if (i != 0 && middle_neuron_num / i == 1) {
      threads.push_back(std::thread(&DenoisingAutoencoder::middleOutThread, this,
                                    std::ref(input), i, middle_neuron_num));
    } else {
      threads.push_back(std::thread(&DenoisingAutoencoder::middleOutThread, this,
                                    std::ref(input), i, i + charge));
    }
  }
  for (std::thread &th : threads) th.join();

  threads.clear();
  if (output_neuron_num <= num_thread) { charge = 1; }
  else { charge = output_neuron_num / num_thread; }
  for (int i = 0; i < output_neuron_num; i += charge) {
    if (i != 0 && output_neuron_num / i == 1) {
      threads.push_back(std::thread(&DenoisingAutoencoder::outOutThread, this, i, output_neuron_num));
    } else {
      threads.push_back(std::thread(&DenoisingAutoencoder::outOutThread, this, i, i + charge));
    }
  }
  for (std::thread &th : threads) th.join();

  if (showResult) {
    for (int neuron = 0; neuron < output_neuron_num; ++neuron) {
      std::cout << "output[" << neuron << "]: " << learnedO[neuron] << " ";
    }
    std::cout << std::endl;
  }

  return learnedO;
}

void DenoisingAutoencoder::middleOutThread(const std::vector<double> in,
                                           const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) learnedH[neuron] = middle_neurons[neuron].output(in);
}

void DenoisingAutoencoder::outOutThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) learnedO[neuron] = output_neurons[neuron].output(learnedH);
}

unsigned long DenoisingAutoencoder::getCurrentMiddleNeuronNum() {
  return middle_neuron_num;
}

double DenoisingAutoencoder::mean_squared_error(double output, double answer) {
  return (output - answer) * (output - answer) / 2;
}