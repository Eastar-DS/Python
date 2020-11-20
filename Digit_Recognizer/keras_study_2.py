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
'''













