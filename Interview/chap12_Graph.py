import collections
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from typing import *

#200. Number of Islands
class Solution200:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(i,j):
            if(i<0 or j<0 or i >= len(grid) or j >= len(grid) or grid[i][j] != '1'):
                return            
            grid[i][j] = 0
            dfs(i-1,j)
            dfs(i,j-1)
            dfs(i+1,j)
            dfs(i,j+1)
            
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if(grid[i][j] == '1'):
                    dfs(i,j)
                    count += 1        
        return count