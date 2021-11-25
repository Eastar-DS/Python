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
Day1 Binary Search : 704 Binary Search, 278 First Bad Version, 35 Search Insert Position
Day2 Two Pointers : 977 Squares of a Sorted Array, 189 Rotate Array
Day3 Two Pointers : 
Day4 Two Pointers : 
Day5 Two Pointers : 


 
"""


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        if(target < nums[0] or nums[-1] < target):
            return -1
        while(left <= right):
            index = (left+right)//2
            if(nums[index] == target):
                return index
            if(nums[index] < target):
                left += 1
            else:
                right -= 1
        return -1
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
    
    
    
        
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left, right = 1, n
        while(left < right):
            index = (left + right)//2
            if(isBadVersion(index)):
                right = index
            else:
                left = index + 1
            
        return left
    



    def searchInsert(self, nums: List[int], target: int) -> int:
        










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



    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left, right = 1, n
        while(left <= right):
            index = (left + right)//2
            if(isBadVersion(index) == True):
                right = index
            else:
                left = index
            
            if(left == right - 1):
                if(isBadVersion(left) == True):
                    return left
                else:
                    return right
                
            if(left == right):
                return right
            """
            만약 첫번째가 bad가 아니면 left는 false, right는 true.
                left +1 == right.
                right를 반환해야함.
            만약 첫번째가 bad면 둘다 true인채로 left +1 == right가됨.
                left를 반환해야함.
            
            만약 n=1이면 바로반환해줘야함.
            """


"Runtime: 40 ms, faster than 98.01%, 솔루션 유료"
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        if(target <= nums[left]):
            return 0
        if(nums[right] < target):
            return right + 1
        """타겟이 리스트의 범위 밖에있으면 먼저 처리
            리스트 원소가 한개여도 이걸로 처리됨."""
        
        while(left < right):
            index = (left + right) // 2
            
            if(nums[index] == target):
                return index
            
            if(left + 1 == right):
                return right
            
            if(nums[index] < target):
                left = index
            else:
                right = index
        
        "남은경우의수는 왼쪽보다 크거나 오른쪽보다 작거나 같음."
            
            
            
            





"""
Given a sorted array of distinct integers and a target value, 
    return the index if the target is found. If not, return the index 
        where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.

Example 1:

Input: nums = [1,3,5,6], target = 5
Output: 2
Example 2:

Input: nums = [1,3,5,6], target = 2
Output: 1
Example 3:

Input: nums = [1,3,5,6], target = 7
Output: 4
Example 4:

Input: nums = [1,3,5,6], target = 0
Output: 0
Example 5:

Input: nums = [1], target = 0
Output: 0
"""



















