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


# train_df.info()
# print('_'*40)
# test_df.info()

# train_df.describe()
# train_df.describe(include=['O'])


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
 train_df.groupby() 만하면 데이터가 안보임. 
 train_df.groupby(['Pclass']).size()
Pclass
1    216
2    184
3    491
dtype: int64
 train_df.groupby(['Pclass']).sum() 
 : Pclass가 1인 데이터들의 다른 label들(Survived) 값을 모두 합함.
        Survived      Age
Pclass                   
1            136  7111.42
2             87  5168.83
3            119  8924.92

 train_df.groupby().mean()
 : Pclass가 1인 데이터들의 다른 label들(Survived) 값들의 평균.
        Survived        Age
Pclass                     
1       0.629630  38.233441
2       0.472826  29.877630
3       0.242363  25.140620

train_df.groupby().sum().sort_values(by='Age', ascending=False)
        Survived      Age
Pclass                   
3            119  8924.92
1            136  7111.42
2             87  5168.83

 '''


#train_df[['Age','Survived']] 에서 나이로 그룹짓고 얼마나 생존했는지 보고싶다.
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=40)

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

'''
나타나는 그래프의 개수 : col x row 개
row를 hue로 바꾸면 한 열의 그래프가 겹쳐져서나옴.
size => height 세로사이즈
aspect 가로사이즈
x축 변수 : map에서 'age'
'''

# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

'''
Numerical data와 nonnumerical data를 연관시키고싶어함.
'''

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


#Feature(Ticket, Cabin) 제거하기
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape



#Title feature 생성하기
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

'''
pd.crosstab(행의특성, 열의특성)
pd.crosstab(train_df['Title'], train_df['Sex'])
Sex       female  male
Title                 
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1
'''
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
'''
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
    Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4    Rare  0.347826

train_df['Title'].value_counts()
Mr        517
Miss      185
Mrs       126
Master     40
Rare       23
'''


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
'''
   PassengerId  Survived  Pclass  ...     Fare Embarked  Title
0            1         0       3  ...   7.2500        S      1
1            2         1       1  ...  71.2833        C      3
2            3         1       3  ...   7.9250        S      2
3            4         1       1  ...  53.1000        S      3
4            5         0       3  ...   8.0500        S      1

map함수는 반드시 series 형태에 써야하므로 dataset['Title']에 사용하게 되는것.
'''

#Name, PassengerID 특성(Feature) 없애기
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
'''
pd.drop()에서 열의 데이터를 삭제할때는 꼭!!! axis=1 또는 axis='columns'
를 입력해주어야함. 
원래 데이터프레임에서는 drop하고싶지 않다면 inplace = True 를 적어주도록하자.
'''

#Categorical data들 Numerical data로 바꾸기
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()





























