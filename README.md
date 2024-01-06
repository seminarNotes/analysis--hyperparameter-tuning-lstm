# 하이퍼파라미터 튜닝 : LSTM 모델  
최초 작성일 : 2024-01-06  
마지막 수정일 : 2024-01-06
  
## 0. Overview
시계열 데이터를 예측하는 LSTM 모델을 구축하고, DataBase(Mysql)을 이용해서, 하이퍼파라미터 튜닝을 시도하였다.

## Table of Contents
1. [Introduction](#1.-Introduction)


## 1. Introduction 

LSTM(Long Short-Term Memory)는 순환 신경망(RNN, Recurrent Neural Network)의 한 종류로, 주로 시계열 데이터나 자연어 처리 등의 문제에 적용되는 딥러닝 모델이다. LSTM은 RNN이 갖는 한계점 중 하나인 '기울기 소실' 문제를 해결하고 정보의 장기 의존성을 효과적으로 학습할 수 있는 장점이 있다.

한편, 하이퍼파라미터 튜닝(Hyper-Parameter Tunning)은 머신러닝 모델의 성능을 최적화 하기 위해 모델의 하이퍼파라미터를 조정하는 프로세스를 의미한다. 하이퍼파라미터는 모델을 학습하는 동안 일정하게 유지되는 파라미터로, 하이퍼파라미터 튜닝 과정을 통해 모델의 성능을 향상시키기 위해 다음과 같은 단계로 최적의 세팅을 찾는 작업이 수행된다.

1. 하이퍼파라미터 정의
2. 탐색 공간 정의
3. 최적화 알고리즘 선택
4. 실행 수행 및 평가
5. 최적의 하이퍼 파라미터 선택 및 최종 학습 모델

하이퍼파라미터 정의는 어떤 하이퍼 파라미터를 최적화 시킬 것인지에 대해 정의하는 단계이다. 대표적인 하이퍼파라미터로는 학습률(learning rate), 배치 크기(batch size), 은닉층 크기(size of hidden layer), 규제화 매개변수(penalty parameter) 등이 있다. 본 실험에서는 에포크(Epoch), 학습률(learning rate), 학습 데이터 비율(ratio of train set), LSTM 모델 내 window 크기(size of sequence), 입력층 크기(size of input), 은닉층 크기(size of hidden), 출력층 크기(size of output)으로 정의하였다.

탐색 공간 정의는 각 하이퍼 파라미터에 대한 탐색 범위를 정의하는 단계이다. 본 실험에서는 다음과 같이 탐색 범위를 정의했으며, 실험은 각 하이퍼 파라미터의 순서쌍(Catesian product)에 대해서 수행 및 평가가 이루어진다.
|Name|Search Space Size|Value|
|:--:|:--:|:--:|
|Epoch|2|20, 30|
|Batch Size|2|10, 20|
|Learning Rate|2|0.001, 0.005|
|Ratio of train set|2|0.9, 0.8|
|size of sequence|1|50|
|size of input|2|50, 64|
|size of hidden|2|64, 128|
|size of output|1|1|

최적화 알고리즘은 하이퍼 파라미터 튜닝 과정에서 어떤 하이퍼 파라미터 조합을 시도할 것인지를 결정하는 방법을 의미하고, 주로 랜덤 서치(Random Search), 그리드 서치(Grid Search), 베이지언 최적화(Baysian Optimization)을 사용하며, 각 방법은 계산 비용과 효율에 대한 장담점이 존재한다. 본 실험에서는 정의한 전체 탐색 곤간을 탐색할 수 있는 그리드 서치 방법을 선택한다.

실행 수행 및 평가는 하이퍼파라미터 공간 내 각 순서쌍에 대해 반복적으로 모델을 학습하고, 평가하여, 성능을 기록하여, 최적의 하이퍼 파라미터를 찾는 과정을 의미한다. 본 실험에서는 DataBase에 Table를 구축해서, 실행 수행 전 탐색 할 하이퍼파라미터를 미리 저장해놓고, 순서쌍에 대해 작업 코드(task code)를 부여하여, 하이퍼 파라미터 순서쌍과 모델의 지표를 관리하였다. 이 후, 예측이 가장 잘 된 모델을 결정하기 위해 실제값과 예측값 사이의 평균 제곱 오차(MSE, Mean Squaere Error)를 계산하여, 가장 작은 MSE를 갖는 작업 코드를 찾는 방식으로 하이퍼파라미터 튜닝을 구성하였다.

$$\hat{h} = \arg\min_{h}\sum_{i \leq n} \left[ \text{Actual Data}(i) - \text{Predicted Data}(i)(h) \right]^2$$


