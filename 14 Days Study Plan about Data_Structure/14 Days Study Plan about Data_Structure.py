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
Day 1 Array : 217. Contains Duplicate, 53. Maximum Subarray
Day 2 Array : 1. Two Sum, 88. Merge Sorted Array
Day 3 Array : 
Day 4 Array : 
Day 5 Array : 
"""

class Solution:
    "One line Solution"
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) != len(set(nums))
        
    

    
    
    
    
    
    
    
class My_Solution:
    "솔루션 유료"
    """Time complexity: O(N lg N), 
                memory: O(1) - not counting the memory used by sort"""
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums.sort()
        index = 0
        while(index < len(nums) - 1):
            if(nums[index] == nums[index + 1]):
                return True
            index += 1
        
        return False

    """
Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

Example 1:
Input: nums = [1,2,3,1]
Output: true

Example 2:
Input: nums = [1,2,3,4]
Output: false

Example 3:
Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true
    """    
    


    
    "솔루션 유료" "https://juneyr.dev/2019-11-21/maximum-subarray"
    def maxSubArray(self, nums: List[int]) -> int:
        "어떻게 해야할지 감이안온다... -> 카데인알고리즘"
        idx = 1
        value = nums[0]
        maximum = value
        while(idx < len(nums)):
            value = max(nums[idx], nums[idx] + value)
            maximum = max(value, maximum)
            idx += 1
        return maximum
    
    """
Example 1:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
Example 2:

Input: nums = [1]
Output: 1
Example 3:

Input: nums = [5,4,-1,7,8]
Output: 23
    """    
    
    
    
    
    
    
    
        
        
        
        
        
        