# -*- coding: utf-8 -*-
"""
Machine learning with Digit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
'matplotlib inline'

'난수 일정하게 만들기'
np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# convert to one-hot-encoding
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


train = pd.read_csv("Python/Digit_Recognizer/train.csv")
test = pd.read_csv("Python/Digit_Recognizer/test.csv")

#Y에 정답, X에 픽셀값들
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)

del train

g = sns.countplot(Y_train)
Y_train.value_counts()

'2.2 Check for null and missing values'

X_train.isnull().any().describe()
test.isnull().any().describe()


'2.3 Normalization'
X_train = X_train / 255.0
test = test / 255.0

'2.3 Reshape'
'데이터를 [0]의 형태(,,,1)로 28x28짜리 행렬로 만들어서 데이터개수(-1,)만큼 펼침.'
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

'2.5 Label encoding'
#숫자 0~9중 하나를 표시하기위해 1을 나타낸다고 치면[0,1,0,0,0,0,0,0,0,0].
Y_train = to_categorical(Y_train, num_classes = 10)


# Set the random seed
random_seed = 2


# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


# Some examples
g = plt.imshow(X_train[0][:,:,0])
for i in range(10) :
    g = plt.imshow(X_train[i][:,:,0])

'사진 여러개 보려면 어떻게 해야할까?'









#사진출력해보기



'''
def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    
    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()
    
    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i+1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels = ['Survived', 'Dead'], autopct = '%1.1f%%')
        
    plt.show()
    
def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
'''
    
    
    