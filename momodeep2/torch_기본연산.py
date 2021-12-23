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

#lec1
m1 = torch.FloatTensor([[1,2],[3,4]])
"""
Broadcasting : 자동적으로 사이즈맞춰서 계산
[[1,2]] + [3] = [[4,5]]
[[1,2]] + [[3],[4]] = [[4,5],[5,6]]

*(.mul) vs .matmul
[[1,2],[3,4]] * [[1],[2]] = [[1,2],[6,8]]
[[1,2],[3,4]].matmul([[1],[2]]) = [[5],[11]]

.mean()
m1 = torch.FloatTensor([[1,2],[3,4]])
m1.mean() = 2.5
m1.mean(dim=0) = [2,3]
m1.mean(dim=1) = [1.5,3.5]
m1.mean(dim=-1) = [1.5,3.5]

.sum()
m1.sum() = 10
m1.sum(dim=0) = 10
m1.sum(dim=1) = [3,7]
m1.sum(dim=-1) = [3,7]

.max()
m1.max() = 4
m1.max(dim=0) = [3,4], [1,1](argmax)
m1.max(dim=0)[0] = [3,4]
m1.max(dim=0)[1] = [1,1]
m1.max(dim=1) = [2,4],[1,1]
m1.max(dim=-1) = [2,4],[1,1]
"""
#lec2
t = np.array([[[0,1,2],
              [3,4,5]],
              [[6,7,8],
               [9,10,11]]])
ft = torch.FlatTensor(t)
"""
.view() : ==Reshape
ft.view([-1,3]) = [[012],[345],[678],[9,10,11]]
ft.view([-1,3]).shape = [4,3]
ft.view([-1,1,3]) = [[[012]]
                     [[345]]
                     [[678]]
                     [[9,10,11]]] : 4개의 batch, batch당1개의데이터, 3개의특성
ft.view([-1,1,3]).shape = [4,1,3]

"""

ft = torch.FloatTensor([[0],[1],[2]])
"""
.squeeze() : 사이즈 1 쥐어짜서 없애줌.
ft.shape = [3,1]
ft.squeeze(dim = 0) = 그대로 => why? 첫번째 사이즈가 1이면 쥐어짜는거임.
ft.squeeze() = [0,1,2]
ft.shape = [3]
"""

ft = torch.FloatTensor([0,1,2])
"""
.unsqueeze()
ft.shape = [3]
ft.unsqueeze(0) = ft.unsqueeze(dim = 0) = [[0,1,2]]
ft.unsqueeze(0).shape = [1,3]

ft.view(1,-1) = [[0,1,2]]

ft.unsqueeze(1) = [[0][1][2]]
"""

lt = torch.LongTensor([1,2,3,4])
bt = torch.ByteTensor([True,False,False,True])
"""
Type Casting
lt.float()
bt.long() = [1,0,0,1]
bt.float() = [1.,0.,0.,1.]
"""

x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[5,6],[7,8]])
"""
Concatenate
.cat()
torch.cat([x,y], dim = 0) = 위아래로 잇는다. (2,4)
torch.cat([x,y], dim = 1) = 옆으로 잇는다. (4,2)
"""

x = torch.FloatTensor([1,4])
y = torch.FloatTensor([2,5])
z = torch.FloatTensor([3,6])
"""
Stacking
.stack()
torch.stack([x,y,z]) = [[1,4],[2,5],[3,6]]
torch.stack([x,y,z], dim = 1) = [[1,2,3],[4,5,6]]

torch.cat([x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0)], dim=0)
"""

x = torch.FloatTensor([[0,1,2],[2,1,0]])
"""
Ones and Zeros
torch.ones_like(x) = [[1,1,1],[1,1,1]]
torch.zeros_like(x) = [[0,0,0],[0,0,0]]

중요한것은 이것들의 device가 같은곳에서 생김. 어떤 연산은 같은 디바이스에 있어야 가능하다!
"""

x = torch.FloatTensor([[1,2],[3,4]])
"""
x.mul(2.) = [[2,4],[6,8]]
x는 변하지않음.
x.mul_(2.) = [[2,4],[6,8]]
x = [[2,4],[6,8]]
x가 변함.
"""














































