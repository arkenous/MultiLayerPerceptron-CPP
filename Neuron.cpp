//
// Created by Kensuke Kosaka on 2016/10/27.
//

#include <iostream>
#include <sstream>
#include "Neuron.h"

/**
 * Neuronのコンストラクタ
 * @param inputNeuronNum 入力ニューロン数（入力データ数）
 * @param dropout_ratio Dropout率
 * @return
 */
Neuron::Neuron(unsigned long inputNeuronNum, int activation_type, double dropout_ratio) {
    this->inputNeuronNum = inputNeuronNum;
    this->activation_type = activation_type;
    this->inputWeights.reserve(this->inputNeuronNum);
    this->dropout_ratio = dropout_ratio;
    std::random_device rnd; // 非決定的乱数生成器
    std::mt19937 mt; // メルセンヌ・ツイスタ
    mt.seed(rnd());
    std::uniform_real_distribution<double> real_rnd(0.0, 1.0);
    this->bias = real_rnd(mt); // バイアスを乱数で設定

    // 結合荷重をを乱数で初期化
    for (int i = 0; i < this->inputNeuronNum; ++i) {
        this->inputWeights.push_back(real_rnd(mt));
    }

    // Adamのイテレーションカウント変数を0で初期化する（learnの最初にインクリメント）
    this->iteration = 0;
    this->m = std::vector<double>(inputNeuronNum, 0.0);
    this->nu = std::vector<double>(inputNeuronNum, 0.0);
    this->m_hat = std::vector<double>(inputNeuronNum, 0.0);
    this->nu_hat = std::vector<double>(inputNeuronNum, 0.0);
}

/**
 * 受け取った0.0以上1.0未満の乱数値からdropout_maskを設定する
 * @param random_value 0.0以上1.0未満の乱数値
 */
void Neuron::dropout(double random_value) {
    if (random_value < dropout_ratio) this->dropout_mask = 0.0;
    else this->dropout_mask = 1.0;
}

/**
 * dropout_maskが1.0であれば，Adamを用いてニューロンの結合荷重を学習し，確率的勾配降下でバイアスを更新する
 * @param delta 損失関数を偏微分したもの（これに一つ前の層の出力データを掛けて傾きを得る）
 * @param inputValues 一つ前の層の出力データ
 */
void Neuron::learn(double delta, std::vector<double> inputValues){
    this->delta = delta;

    // Adamを用いて重み付けを学習する
    if (this->dropout_mask == 1.0) {
        this->iteration += 1;
        for (int i = 0; i < this->inputNeuronNum; ++i) {
            this->m[i] = this->beta_one * this->m[i] + (1 - this->beta_one) * (this->delta * inputValues[i]);
            this->nu[i] = this->beta_two * this->nu[i] + (1 - this->beta_two) * pow((this->delta * inputValues[i]), 2);
            this->m_hat[i] = this->m[i] / (1 - pow(this->beta_one, this->iteration));
            this->nu_hat[i] = sqrt(this->nu[i] / (1 - pow(this->beta_two, this->iteration))) + this->epsilon;
            this->inputWeights[i] -= this->alpha * (this->m_hat[i] / this->nu_hat[i]);
        }

        // 確率的勾配降下でバイアスを更新
        this->bias -= (this->alpha * this->delta) - (this->alpha * this->rambda * this->bias);
    }
}

/**
 * ニューロンの出力メソッド．バイアスや重み付けにdropout_ratioを掛けて処理する
 * @param inputValues 一つ前の層の出力データ
 * @return ニューロンの出力値（活性化関数より得られた値）
 */
double Neuron::output(std::vector<double> inputValues) {
    double sum = this->bias * this->dropout_ratio;
    for (int i = 0; i < this->inputNeuronNum; ++i) {
        sum += inputValues[i] * (this->inputWeights[i] * this->dropout_ratio);
    }

    double activated;
    if (activation_type == 0) activated = activation_identity(sum);
    else if (activation_type == 1) activated = activation_sigmoid(sum);
    else if (activation_type == 2) activated = activation_tanh(sum);
    else activated = activation_relu(sum);

    return activated;
}

/**
 * ニューロンの出力を得て，それにdropout_maskを掛ける
 * @param inputValues ニューロンの入力データ
 * @return ニューロンの出力
 */
double Neuron::learn_output(std::vector<double> inputValues){
    // 入力側の細胞出力の重み付き和をとる
    double sum = this->bias;
    for (int i = 0; i < this->inputNeuronNum; ++i) {
        sum += inputValues[i] * this->inputWeights[i];
    }

    // 得られた重み付き和を活性化関数に入れて出力を得る
    double activated;
    if (activation_type == 0) activated = activation_identity(sum);
    else if (activation_type == 1) activated = activation_sigmoid(sum);
    else if (activation_type == 2) activated = activation_tanh(sum);
    else activated = activation_relu(sum);

    return activated * this->dropout_mask;
}

/**
 * 活性化関数：恒等写像
 * @param x 入力
 * @return 計算結果
 */
double Neuron::activation_identity(double x) {
    return x;
}

/**
 * 活性化関数 : シグモイド関数
 * @param x 入力
 * @return 計算結果
 */
double Neuron::activation_sigmoid(double x){
    return 1.0 / (1.0 + pow(M_E, -x));
}

/**
 * 活性化関数 : tanh
 * @param x 入力
 * @return 計算結果
 */
double Neuron::activation_tanh(double x) {
    return std::tanh(x);
}

/**
 * 活性化関数 : ランプ関数（ReLU）
 * @param x 入力
 * @return 計算結果
 */
double Neuron::activation_relu(double x) {
    return std::max(0.0, x);
}

/**
 * このニューロンの指定された入力インデックスの結合荷重を返す
 * @param i 入力インデックス
 * @return 結合荷重
 */
double Neuron::getInputWeightIndexOf(int i){
    return this->inputWeights[i];
}

/**
 * 現在の修正量を返す
 * @return 修正量
 */
double Neuron::getDelta() {
    return this->delta;
}

/**
 * このニューロンの結合荷重を文字列でまとめて返す
 * @return このニューロンの結合荷重をまとめた文字列
 */
std::string Neuron::toString() {
    std::stringstream ss;
    ss << "weight : ";
    for (int neuron = 0; neuron < inputNeuronNum; ++neuron) {
        ss << inputWeights[neuron] << " , ";
    }

    std::string output = ss.str();
    return output;
}
