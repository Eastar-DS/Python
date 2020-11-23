# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:07:40 2020

@author: user
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd





fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape, y_train_full.shape, X_test.shape, y_test.shape)

print(X_train_full.dtype, y_train_full.dtype, X_test.dtype, y_test.dtype)
'''
data type : uint8 , array 형식
'''

#픽셀값을 0~1사이값으로 수정

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:]
X_test = X_test / 255.0

plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()

'''
X_train_full[1:].shape
(59999, 28, 28)

X_train_full[:1].shape
(1, 28, 28)

X_train_full[:1] == X_train_full[0]

valid : 앞에 5000개
train : 뒤에 55000개
train_full, test : 60000개
'''

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_names[y_train[0]]


#목록확인하기

# Where to save the figures
# PROJECT_ROOT_DIR = "."
# CHAPTER_ID = "ann"
# IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
# os.makedirs(IMAGES_PATH, exist_ok=True)

# def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#     path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#     print("Saving figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi=resolution)


# n_rows = 4
# n_cols = 10
# plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
# for row in range(n_rows):
#     for col in range(n_cols):
#         index = n_cols * row + col
#         plt.subplot(n_rows, n_cols, index + 1)
#         plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
#         plt.axis('off')
#         plt.title(class_names[y_train[index]], fontsize=12)
# plt.subplots_adjust(wspace=0.2, hspace=0.5)
# save_fig('fashion_mnist_plot', tight_layout=False)
# plt.show()


#시퀀셜 API를 사용하여 모델 만들기
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

'''

☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★

첫 번째 라인은 Sequential 모델을 만듭니다. 이 모델은 가장 간단한 케라스의 신경망 모델입니다.
순서대로 연결된 층을 일렬로 쌓아서 구성합니다. 이를 Sequential API라고 부릅니다.

그다음 첫 번째 층을 만들고 모델에 추가합니다. Flatten 층은 입력 이미지를 1D 배열로 변환합니다.
즉, 입력 데이터 X를 받으면 X.reshape(-1,1)를 계산합니다. 
이 층은 어떤 모델 파라미터도 가지지 않고 간단한 전처리를 수행할 뿐입니다.
모델의 첫 번째 층이므로 input_shape를 지정해야 합니다. 여기에는 배치 크기를 제외하고 샘플의 크기만 써야 합니다.
또는 첫 번째 층으로 input_shape=[28,28]로 지정된 keras.layers.InputLayer 층을 추가할 수도 있습니다.

그다음 뉴런 300개를 가진 Dense 은닉층을 추가합니다. 이 층은 reLU 활성화 함수를 사용합니다.
Dense 층마다 각자 가중치 행렬을 관리합니다. 이 행렬에는 층의 뉴런과 입력 사이의 모든 연결 가중치가 포함됩니다.
또한 (뉴런마다 하나씩 있는) 편향도 벡터로 관리합니다.
이 층은 입력 데이터를 받으면[식 10-2]를 계산합니다.

다음 뉴런 100개를 가진 두 번째 Dense 은닉층을 추가합니다. 역시 ReLU 활성화 함수를 사용합니다.

마지막으로 (클래스마다 하나씩) 뉴런 10개를 가진 Dense 출력층을 추가합니다. 
(배타적인 클래스이므로) 소프트맥스 활성화 함수를 사용합니다.
0~9(10개)중 하나로 분류해야하므로 뉴런10개와 소프트맥스함수를 사용!

'''
# a = np.array([[[1,2,3,4,5,6,7,8,9,0],
#               [1,2,3,4,5,6,7,8,9,0]]
#       ])
# a.reshape(-1,1)
'''
array([[1],
       [2],
       [3],
       [4],
       [5],
       [6],
       [7],
       [8],
       [9],
       [0],
       [1],
       [2],
       [3],
       [4],
       [5],
       [6],
       [7],
       [8],
       [9],
       [0]])
'''
# b = np.array([[[1,2,3,4],
#               [1,2,3,4]],
#               [[1,2,3,4],
#               [1,2,3,4]]
#       ])
# b.reshape(-1,1)
'''
array([[1],
       [2],
       [3],
       [4],
       [1],
       [2],
       [3],
       [4],
       [1],
       [2],
       [3],
       [4],
       [1],
       [2],
       [3],
       [4]])

reshape(1,-1) 을 사용할경우에는 한행에 모든 데이터가 들어가므로 값 하나씩 따로 설정이 안됨.

'''
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.layers
model.summary()

'''
Keras의 다양한 사용법
from keras.layers import Dense
output_layer = Dense(10)

from tensorflow.keras.layers import Dense
output_layer = Dense(10)

from tensorflow import keras
output_layer = keras.layers.Dense(10)

from tensorflow.keras import layers
output_layer = layers.Dense(10)
'''

hidden1 = model.layers[1]
hidden1.name
model.get_layer(hidden1.name) is hidden1
weights, biases = hidden1.get_weights()
weights
weights.shape
biases
biases.shape

'''
Dense는 bias를 0으로 초기화함. 다른 초기화 방법을 사용하고싶다면 층을 만들때 
kernel_initializer(커널은 연결 가중치 행렬의 또 다른 이름.) 와 
bias_initializer 매개변수를 설정할 수 있음. 11장에서 초기화 방법에 대해 알아볼것임.
'''
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
'''
이코드와 같음.
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=[keras.metrics.sparse_categorical_accuracy])

레이블이 정수 하나로 이루어져 있고(즉, 샘플마다 타깃 클래스 인덱스 하나가 있습니다. 여기서는 0~9까지 정수) 
클래스가 배타적이므로 "sparse_categorical_crossentropy" 손실을 사용합니다. 만약 샘플마다 클래스 별 타깃 확률을
가지고 있다면 (예를들어 클래스 3의경우 [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]인 원-핫 벡터라면 one-hot)
대신 "categorical_crossentropy" 손실을 사용해야 합니다. (하나 또는 그 이상의 이진 레이블을 가진) 이진 분류를
수행한다면 출력층에 "softmax"함수 대신 "sigmoid" 함수를 사용하고 "binary_crossentropy" 손실을 사용합니다.

TIP : 희소한 레이블(sparse label즉 클래스 인덱스)을 원-핫 벡터 레이블로 변환하려면 
keras.utils.to_categorical()함수를 사용하세요. 그 반대로 변환하려면 axis=1로 지정하여 np.argmax() 함수를 사용합니다.

옵티마이저에 "sgd"를 지정하면 기본 확률적 경사 하강법(stochastic gradient descent)을 사용하여 모델을 훈련한다는 의미.
다른 말로 하면 케라스가 앞서 소개한 역전파 알고리즘을 수행합니다(즉, 후진 모드 자동 미분과 경사 하강법). 
11장에서 더 효율적인 옵티마이저를 설명합니다.

SGD 옵티마이저를 사용할 때 학습률을 튜닝하는 것이 중요. optimizer=keras.optimizers.SGD(lr= ??? ).
기본값은 lr= 0.01
'''

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

#모델 훈련시킬때는 간단하게 fit()메서드를 호출합니다.
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

'''
validation_data 매개변수에 검증 세트를 전달하는 대신 케라스가 검증에 사용할 훈련 세트의 비율을 지정할 수 있습니다.
예를 들어 validataion_split=0.1로 쓰면 케라스는 검증에 (섞기 전의) 데이터의 마지막 10%를 사용합니다.

훈련 데이터와 검증 데이터의 크기가 맞지 않으면 예외가 발생됩니다. 이 메세지는 아주 명확합니다.
예를들어 일렬로 펼친 이미지(X_train.reshape(-1,784))를 담은 배열로 이 모델을 훈련한다면 다음과 같은 예외가 발생.
"ValueError: Error when checking input: expected flatten_input to have 3 dimensions, 
but got array with shape(60000, 784)."

어떤 클래스는 많이 등장하고 다른 클래스는 조금 등장하여 훈련 세트가 편중되어 있다면 fit() 메서드를 호출할 때
class_weight 매개변수를 지정하는 것이 좋습니다. 적게 등장하는 클래스는 높은 가중치를 부여하고 많이 등장하는 클래스는
낮은 가중치를 부여합니다. 케라스가 손실을 계산할 때 이 가중치를 사용합니다. 

샘플별로 가중치를 부여하고 싶다면 sample_weight 매개변수를 지정합니다(class_weight와 sample_weight가 모두 지정되면 
                                                                                    두 값을 곱하여 사용합니다.)

fit() 메서드가 반환하는 History 객체에는 훈련파라미터(history.params), 수행된 에포크리스트(history.epoch)가 포함.
이 객체의 가장 중요한 속성은 에포크가 끝날 때마다 훈련 세트와 (있다면) 검증 세트에 대한 손실과 측정한 지표를 담은
딕셔너리(history.history)입니다.

이 딕셔너리를 사용해 판다스 데이터프레임을 만들고 plot() 메서드를 호출하면 학습곡선(Learning curve)을 볼 수 있습니다.
'''

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
#save_fig("keras_learning_curves_plot")
plt.show()

'''
케라스에서는 fit()메서드를 다시 호출하면 중지되었던 곳에서부터 훈련을 이어갈 수 있습니다.
(89% 검증 정확도에 가까이 도달할 것임.)

모델 성능이 만족스럽지 않다면 처음으로 되돌아가서 하이퍼파라미터를 튜닝해야 합니다. 맨처음 확인할 것은 학습률(learning rate)입니다.
학습률이 도움이 되지 않으면 다른 옵티마이저를 테스트해보세요.
(항상 다른 하이퍼파라미터를 바꾼 후에는 학습률을 다시 튜닝해야 합니다.)
여전히 성능이 높지 않으면 층 개수, 층에 있는 뉴런 개수, 은닉층이 사용하는 활성화 함수와 같은 모델의 하이퍼파라미터를 튜닝.
배치 크기와 같은 다른 하이퍼파라미터를 튜닝해볼 수도있음.(fit() 메서드를 호출할때 batch_size 매개변수로 지정. 기본값은 32)
끝에서 하이퍼파라미터 튜닝에 대해 다시 알아볼것임.
모델의 검증 정확도가 만족스럽다면 모델을 상용 환경으로 배포하기 전에 테스트 세트로 모델을 평가하여 일반화 오차를 추정해야합니다.
이때 evaluate() 메서드를 사용.(이 메서드는 batch_size와 sample_weight 같은 다른 매개변수도 지원.)
'''

model.evaluate(X_test, y_test)


X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)


y_pred = model.predict_classes(X_new)
y_pred
np.array(class_names)[y_pred]

y_new = y_test[:3]
y_new

plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[y_test[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
#save_fig('fashion_mnist_images_plot', tight_layout=False)
plt.show()









































































