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









































