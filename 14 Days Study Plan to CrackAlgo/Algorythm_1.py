import collections
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from typing import *
"""
Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4
Example 2:

Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
 

Constraints:

1 <= nums.length <= 10^4
-10^4 < nums[i], target < 10^4
All the integers in nums are unique.
nums is sorted in ascending order.
"""
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        index = -1
        if(target < nums[0] or nums[-1] < target):
            return index
        for num in nums:
            index += 1
            if(target == num):
                return index
        return -1
    

    














class My_Solution:
    def search(self, nums: List[int], target: int) -> int:
        index = -1
        if(target < nums[0] or nums[-1] < target):
            return index
        for num in nums:
            index += 1
            if(target == num):
                return index
        return -1























