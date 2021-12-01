import collections
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from typing import *

class Solution:
    #12/01 : 198. House Robber
    def rob(self, nums: List[int]) -> int:
        """
        Input: nums = [1,2,3,1]
        Output: 4
        
        Input: nums = [2,7,9,3,1]
        Output: 12
        
        input : [1,0,1,0,0,1,0,1]
        output : 4
        """
        """
        디스커스
        rob(i) = Math.max( rob(i - 2) + currentHouseValue, rob(i - 1) )
        
        [183,219,57,193,94,233,202,154,65,240,97,234,100,249,186,66,90,238,168,128,177,235,50,81,
         185,165,217,207,88,80,112,78,135,62,228,247,211]
        """
        last, now = 0, 0
        
        for i in nums: last, now = now, max(last + i, now)
                
        return now