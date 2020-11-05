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
train_df.describe(include=['O'])


'''
train_df['Fare'].describe() 원하는데이터 describe사용하기
describe 사용시 top : 가장 많이나온 데이터, freq  : top이 나온 빈도수
 include = 'all' 사용시  NaN이 나오는 이유는 number형과 object형을 
 동시에 나타내려고 하니 한쪽에서 표시가 안됨.
 include = 'O' 알파벳 O임. 오브젝트형만 가져와서 표시하라는것. 
 그래서 통계적 수치가 없구나!
 
 train_df에서 원하는 목차만 보는법 : train_df[] 속에 ['항목1','항목2']
 의 형태로 쓰면됨.
'''





train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False)\
    .mean().sort_values(by='Survived', ascending=False)

train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False)\
    .mean().sort_values(by='Survived', ascending=False)
    
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False)\
    .mean().sort_values(by='Survived', ascending=False)    
    
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False)\
    .mean().sort_values(by='Survived', ascending=False)






    
'''
 train_df.group_by() 만하면 데이터가 안보임. 
 train_df.group_by(['Pclass']).size()
Pclass
1    216
2    184
3    491
dtype: int64
 train_df.group_by(['Pclass']).sum() 
 : Pclass가 1인 데이터들의 다른 label들(Survived) 값을 모두 합함.
        Survived      Age
Pclass                   
1            136  7111.42
2             87  5168.83
3            119  8924.92

 train_df.group_by().mean()
 : Pclass가 1인 데이터들의 다른 label들(Survived) 값들의 평균.
        Survived        Age
Pclass                     
1       0.629630  38.233441
2       0.472826  29.877630
3       0.242363  25.140620

train_df.group_by().sum().sort_values(by='Age', ascending=False)
        Survived      Age
Pclass                   
3            119  8924.92
1            136  7111.42
2             87  5168.83

 '''










