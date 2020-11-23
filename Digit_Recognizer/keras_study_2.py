# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:07:48 2020

@author: user
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


#10.2.3 Regression MLP
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler










#10.2.3. Regression MLP. 시퀀셜 API를 이용하여 회귀용 다층 퍼셉트론 만들기
housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
'''
Scaler : 기존 변수에 범위를 정규 분포로 변환. 각 피처의 평균을 0, 분산을 1로 변경. 모든 특성들이 같은 스케일을 갖게 됨.
         이상치(Outlier)가 있는 경우 균형 잡힌 척도를 보장할 수 없다는 단점.
'''

np.random.seed(42)
tf.random.set_seed(42)


model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

'왜 val_accuracy가 안나오지?'

plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

y_pred



#10.2.4. Functional API(함수형 API를 사용해 복잡한 모델 만들기)
# input_ = keras.layers.Input(shape=X_train.shape[1:])
# hidden1 = keras.layers.Dense(30, activation="relu")(input_)
# hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# concat = keras.layers.concatenate([input_, hidden2])
# output = keras.layers.Dense(1)(concat)
# model = keras.models.Model(inputs=[input_], outputs=[output])

# model.summary()

'''
먼저 Input 객체를 만들어야 합니다. 이 객체는 shape과 dtype을 포함하여 모델의 입력을 정의합니다.
한 모델은 여러개의 입력을 가질 수 있습니다.

30개의 뉴런과 ReLU 활성화 함수를 가진 Dense 층을 만듭니다. 이 층은 만들어지자마자 입력과 함게 함수처럼 호출됩니다.
이를 함수형 API라고 부르는 이유입니다. 케라스에 층이 연결될 방법을 알려주었을뿐 아직 어떤 데이터도 처리하지 않았습니다.

두 번째 은닉층을 만들고 함수처럼 호출합니다. 첫 번째 층의 출력을 전달한 점을 눈여겨보세요.

Concatenate 층을 만들고 또 다시 함수처럼 호출하여 두 번째 은닉층의 출력과 입력을 연결합니다.
keras.layers.concatenate() 함수를 사용할 수도 있습니다. 이 함수는 Concatenate 층을 만들고 주어진 입력으로 바로 호출합니다.

하나의 뉴런과 활성화 함수가 없는 출력층을 만들고 Concatenate 층이 만든 결과를 사용해 호출합니다. 

마지막으로 사용할 입력과 출력을 지정하여 케라스 Model을 만듭니다.
'''



#10.3 신경망 하이퍼파라미터 튜닝하기

'''
신경망의 유연성은 단점이기도함. 조정할 하이퍼파라미터가 많기 때문. 
간단한 다층 퍼셉트론에서도 층의 개수,  층마다 있는 뉴런의 개수, 각 층에서 사용하는 활성화 함수, 
가중치 초기화 전략 등 많은 것을 바꿀 수 있음.
어떤 하이퍼파라미터 조합이 주어진 문제에 최적인지 알 수 있을까?

한 가지 방법은 많은 하이퍼파라미터 조합을 시도해보고 어떤 것이 검증 세트에서 (또는 K-폴드 교차 검증으로)
가장 좋은 점수를 내는지 확인하는 것.
2장에서 했던것처럼 GridSearchCV나 RandomizedSearchCV를 사용해 하이퍼파라미터 공간을 탐색할 수 있음.
이렇게 하려면 케라스 모델을 사이킷런 추정기처럼 보이도록 바꾸어야 함. 
일련의 하이퍼파라미터로 케라스 모델을 만들고 컴파일하는 함수를 만들어봅시다.
'''
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

keras_reg.fit(X_train, y_train, epochs=100,
              validation_data=(X_valid, y_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])


from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])


rnd_search_cv.best_params_
rnd_search_cv.best_score_
# rnd_search_cv.best_estimator_
# rnd_search_cv.score(X_test, y_test)
'''
추천하는 다른 하이퍼파라미터 최적화 파이썬 라이브러리
Hyperopt / Hyperas, kopt, Talos / Keras Tuner / Scikit-Optimize(skopt) / Spearmint / Hyperband / Sklearn-Deap

[은닉층 개수]
은닉층이 하나인 다층 퍼셉트론이더라도 뉴런개수가 무한하면 아주 복잡한 함수도 모델링이 가능하지만
심층구조가 유리하다!
얼굴인식을위해 첫층은 점,선 두번째층은 사각형,원 고층은 얼굴인식담당.
헤어스타일 인식을위해 저층을 그대로 사용하고 가중치와 편향을 같은값으로 초기화해두면 저수준 구조를 학습할 필요가 없게됨.
이를 전이학습(Transfer Learning)이라고함.



[은닉층의 뉴런 개수]
입력층과 출력층의 뉴런 개수는 해당 작업에 필요한 입력과 출력의 형태에 따라 결정.
MNIST는 28x28 = 784개의 입력뉴런과 10개의 출력뉴런.

은닉층의 구성 방식은 일반적으로 각 층의 뉴런을 점점 줄여서 깔때기처럼 구성.
저수준의 많은 특성이 고수준의 적은 특성으로 합쳐질 수 있기 때문.
ex) 전형적인 MNIST신경망은 첫번째 300, 두 번째 200, 세 번째 100개의 뉴런으로 구성된 세개의 은닉층을 가짐.
하지만 요즘엔 일반적이지 않은 구성임. 
대부분의 경우 모든 은닉층에 같은 크기를 사용해도 동일하거나 더 나은 성능을 냄.
또한 튜닝할 하이퍼파라미터가 층마다 한 개씩이 아니라 전체를 통틀어 한 개가 됨. 
데이터셋에 따라 다르지만 다른 은닉층보다 첫 번째 은닉츠을 크게 하는 것이 도움이 됨.

층의 개수와 마찬가지로 네트워크가 과대적합이 시작되기 전까지 점진적으로 뉴런 수를 늘릴 수 있음.
하지만 실전에서는 필요한 것보다 더 많은 층과 뉴런을 가진 모델을 선택하고, 그런 다음 과대적합되지 않도록 조기 조욜나 규제 기법을
사용하는 것이 간단하고 효과적. = 스트레치 팬츠(Stretch pants)방식 (딱맞는 사이즈바지를 찾지말고 큰거사서 줄여라.)

일반적으로 층의 뉴런 수보다 층 수를 늘리는 쪽이 이득이 많음.


[학습률]
논쟁의 여지 없이 가장 중요한 하이퍼파라미터.
일반적으로 최적의 학습률은 최대학습률(훈련 알고리즘이 발산하는 학습률)의 절반 정도.
좋은 학습률을 찾기위해 매우 낮은 학습률 (10^-5)에서 시작해서 매우 큰 학습률(10)까지 수백 번 반복하여 모델을 훈련.
반복마다 일정한 값을 학습률에 곱함.

[배치크기]
가능한 큰 배치가 좋지만 훈련이 불안정하거나 최종 성능이 만족스럽지 못하면 작은 배치 크기를 사용.
'''













