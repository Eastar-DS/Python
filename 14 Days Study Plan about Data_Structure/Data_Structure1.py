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
Day 3 Array : 350. Intersection of Two Arrays II, 121. Best Time to Buy and Sell Stock
Day 4 Array : 
Day 5 Array : 
"""

class Solution:
    "One line Solution"
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) != len(set(nums))
        
    

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for index, num in enumerate(nums):
           if((target - num) in dic):
               return [dic[target-num], index]
           else:
               dic[num] = index
    
    
    def merge(self, nums1, m, nums2, n):
        while m > 0 and n > 0:
            if nums1[m-1] >= nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class MySolution:
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
    
    
    
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        "[1,2,3,4], 5 -> [1,2]"
        n = len(nums)
        for i in range(n-1):
            for j in range(i+1,n):
                if(nums[i] + nums[j] == target):
                    return [i,j]
        
        
        
        
        
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
        Output: [1,2,2,3,5,6]
        
        Input: nums1 = [1], m = 1, nums2 = [], n = 0
        Output: [1]
        
        """
        nums1[:] = sorted(nums1[:m] + nums2[:n])



    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        Input: nums1 = [1,2,2,1], nums2 = [2,2]
        Output: [2,2]
        
        Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
        Output: [4,9]
        """
        











        