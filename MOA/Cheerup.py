# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:04:43 2020

MOA

Target sparsity
"Percentage of non-zero target class values: 0.343%"\
"Percentage of non-zero non-scored target class values: 0.052%"

Train_Feature
## Number of rows: 23814; Number of columns 876
## Number of "g-" features: 772; Number of "c-" features: 100


"""

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


train_f = pd.read_csv('../MOA/train_features.csv')
train_t = pd.read_csv('../MOA/train_targets_scored.csv')
#test_df = pd.read_csv('test.csv')

train_f.info()


















