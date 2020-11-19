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




























