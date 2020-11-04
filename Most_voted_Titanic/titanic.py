# -*- coding: utf-8 -*-

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
'%matplotlib inline'

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

print(train_df.columns.values)
train_df.head()
train_df.tail()

train_df.info()
print('_'*40)
test_df.info()

train_df.describe()
'''
train_df['Fare'].describe() 원하는데이터 describe사용하기
describe 사용시 top : 가장 많이나온 데이터, freq  : top이 나온 빈도수
 include = 'all' 사용시  NaN이 나오는 이유는 number형과 object형을 
 동시에 나타내려고 하니 한쪽에서 표시가 안됨.
 '''










