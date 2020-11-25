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


train_df = pd.read_csv('Python/Most_voted_Titanic/train.csv')
test_df = pd.read_csv('Python/Most_voted_Titanic/test.csv')
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



#-------------------------------------------------------------------------------------
#데이터 전처리시작!

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
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(float)

train_df.head()


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


#null data 채워넣기
guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            #median값을 구하기위해서 null값을 없애줌
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(float)

train_df.head()
'''
Pclass와 Sex가 같은 데이터들을 모아 Age값의 평균을 뽑아서 df를 만든다.(2,3인 이유는 성별2, Pclass3)
for문을 이용해 Pclass와 Sex가 같은 데이터들중 손실된 Age값에 위에서만든 df에 있는값을 넣어준다.
'''

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False)\
    .mean().sort_values(by='AgeBand', ascending=True)
'''
            AgeBand  Survived
0    (0.34, 16.336]  0.550000
1  (16.336, 32.252]  0.369942
2  (32.252, 48.168]  0.404255
3  (48.168, 64.084]  0.434783
4    (64.084, 80.0]  0.090909

동일한 범위의 5개구간으로 나뉨.
qcut() 사용시 동일하지 않은 범위로 같은 개수의 데이터를 묶어서 나눔.
'''

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()

#SibSp 나 Parch는 동승자이므로 두특성을 합쳐버리자.
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False)\
    .mean().sort_values(by='Survived', ascending=False)


#IsAlone 만들기
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()


#Age * Pclass 굳이 합쳐야?
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


#Embarked의 손실된 데이터는 2개인가 3개밖에안되니까 그냥 가장 많은 빈도수항구로 넣어주자
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#categorical -> numeric
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(float)

train_df.head()


#test_df에 fare가 하나 손실되어있음.
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(float)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df['Pclass'] = train_df['Pclass'].astype(float)
train_df['Title'] = train_df['Title'].astype(float)
train_df['IsAlone'] = train_df['IsAlone'].astype(float)

test_df['Pclass'] = test_df['Pclass'].astype(float)
test_df['Title'] = test_df['Title'].astype(float)
test_df['IsAlone'] = test_df['IsAlone'].astype(float)

feature_name = ['Pclass', 'Age', 'Fare', 'Embarked', 'Title', 'Age*Class']

for dataset in combine:
    for name in feature_name:
        dataset[name] = dataset[name]/max(dataset[name])
        
train_df.head(10)
test_df.head(10)

#데이터 전처리 끝!
#------------------------------------------------------------------------------------

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

X_valid, X_train_ = X_train[:99] , X_train[99:]
y_valid, y_train_ = Y_train[:99] , Y_train[99:]

X_train.head()
Y_train.head()
X_test.head()

#Keras
import tensorflow as tf
from tensorflow import keras
import os

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[8]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
# 생존여부가 0,1 두개이므로 sigmoid로 활성화.
# binary_crossentropy가 손실함수여서그런지 마지막 레이어의 뉴런수를 1개로 해야하네?



model.layers
model.summary()

hidden1 = model.layers[1]
hidden1.name
model.get_layer(hidden1.name) is hidden1
weights, biases = hidden1.get_weights()
weights
weights.shape
biases
biases.shape

model.compile(loss="binary_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
'위와같은 이유로 손실함수는 binary_crossentropy'

#array형식으로 만들어주기
X_train_array = np.array(X_train_)
y_train_array = np.array(y_train_)
X_valid_array = np.array(X_valid)
y_valid_array = np.array(y_valid)
X_test_array = np.array(X_test)



history = model.fit(X_train_array, y_train_array, epochs=20, batch_size = 1, 
                    validation_data=(X_valid_array, y_valid_array))



pd.DataFrame(history.history).plot(figsize=(8, 5))


y_pred = model.predict_classes(X_test)
y_pred



#LogisticRegression
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# acc_log

'''
#coefficient 확인
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

#SVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

#KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest



models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

'''

#제출준비
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_pred[:,0]
    })

submission.to_csv('submission_Keras2.csv', index=False)













