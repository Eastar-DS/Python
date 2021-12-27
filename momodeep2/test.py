import collections
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from typing import *

import numpy as np
import torch 

import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
training_epochs = 15    
batch_size = 100    

mnist_train = dsets.MNIST(root="MNIST_data/", train = True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root="MNIST_data/", train = False, transform=transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(dataset = mnist_train, batch_size = batch_size, shuffle=True, drop_last=True)
"""
DataLoader : 어떤데이터를 로드할거냐
batch_size : 데이터를 몇개씩 잘라서 불러올래
shuffle : 순서를 섞어서 불러올래
drop_last : 마지막 어케할거야
"""
linear = torch.nn.Linear(784,10,bias=True).to(device)
"""
torch.nn 패키지에있는 리니어레이어하나를 사용할거다. 
인풋784 아웃풋10
"""
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
"""
소프트맥스 클래시파이어에 대해서 크로스엔트로피코스트를 사용할거고, 
파이토치에서는 크로스엔트로피로스가 소프트맥스를 자동으로 계산하기때문에 따로선언하지않음.

linear.parameters() = W,b
"""

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)
    
    #데이터로더만큼만 데이터와 레이블을 가져오겠다.
    for X,Y in data_loader:
        X = X.view(-1,28*28).to(device) #[batch_size by 784]
        Y = Y.to(device)
        
        optimizer.zero_grad()
        hypothesis = linear(X) #X를 linear에 넣어서 결과산출
        cost = criterion(hypothesis, Y) #나온 결과와 실제정답 Y로 크로스엔트로피를 계산해 코스트구함
        cost.backward() #백워드를 이용해서 그레디언트를 계산.
        optimizer.step() # 그 그레디언트로 업데이트를 진행.
        
        avg_cost += cost/ total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))


#Test
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1,28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    
    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
    
    #Get one and predict
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r+1].view(-1,28*28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r+1].to(device)
    
    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction ', torch.argmax(single_prediction, 1).item())
    
    plt.imshow(mnist_test.test_data[r:r+1].view(28,28), cmap="Greys", interpolation='nearest')
    plt.show()





































