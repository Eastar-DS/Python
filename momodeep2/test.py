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
m1.max(dim=)
"""































