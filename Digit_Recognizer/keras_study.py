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

















































