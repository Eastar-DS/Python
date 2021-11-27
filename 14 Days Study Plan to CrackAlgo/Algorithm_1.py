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
Day1 Binary Search : 704. Binary Search, 278. First Bad Version, 35. Search Insert Position
Day2 Two Pointers : 977. Squares of a Sorted Array, 189. Rotate Array
Day3 Two Pointers : 
Day4 Two Pointers : 
Day5 Two Pointers : 


 
"""


class Solution:
#day1
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



#day2
    "이개념 익히고싶다."
    """
class SquaresIterator(object):
    def __init__(self, sorted_array):
        self.sorted_array = sorted_array
        self.left_pointer = 0
        self.right_pointer = len(sorted_array) - 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.left_pointer > self.right_pointer:
            raise StopIteration
        left_square = self.sorted_array[self.left_pointer] ** 2
        right_square = self.sorted_array[self.right_pointer] ** 2
        if left_square > right_square:
            self.left_pointer += 1
            return left_square
        else:
            self.right_pointer -= 1
            return right_square
        
    def sortedSquares(self, nums: List[int]) -> List[int]:
        return_array = [0] * len(nums)
        write_pointer = len(nums) - 1
        for square in SquaresIterator(nums):
            return_array[write_pointer] = square
            write_pointer -= 1
        return return_array
    """



    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        kk = k % n
        if(kk > 0):
            nums[:] = nums[n-kk:n] + nums[:n-kk]



    def rotate2(self, nums: List[int], k: int) -> None:
        """
        이게 왜 더빠른지 정말 신기하네...
        """
        def numReverse(start, end):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
        k, n = k % len(nums), len(nums)
        if k:
            numReverse(0, n - 1)
            numReverse(0, k - 1)
            numReverse(k, n - 1)  
















































class MySolution:
    def search(self, nums: List[int], target: int) -> int:
        index = -1
        if(target < nums[0] or nums[-1] < target):
            return index
        for num in nums:
            index += 1
            if(target == num):
                return index
        return -1

    "다른문제 풀고 다시푸니까 런타임 좋아짐. 이게 logN이지~~"
    def search2(self, nums: List[int], target: int) -> int:
        left, right = 0,len(nums)-1        
        index = right // 2
        if(target < nums[0] or nums[-1] < target):
            return -1
        while(index > left):            
            if(nums[index] == target):
                return index
            if(nums[index] < target):
                left = index
            else:
                right = index                
            index = (left + right) // 2
            
        if(target == nums[right]):
            return right
        if(target == nums[left]):
            return left
        else:
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



#day2
    "엄청느리네"
    def sortedSquares(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums) - 1
        leftval, rightval = nums[left]**2, nums[right]**2
        output = []
        while(left <= right):
            if(leftval < rightval):
                output = [rightval] + output
                right -= 1
                rightval = nums[right]**2
            else:
                output = [leftval] + output
                left += 1
                if(left<len(nums)):
                    leftval = nums[left]**2
        return output




    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        kk = k % len(nums)
        while(kk > 0):
            kk -= 1
            element = nums.pop()
            nums.insert(0,element)



    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        Note that you must do this in-place without making a copy of the array.
        """
        index = 0
        for i in range(len(nums)):
            if(nums[index] == 0):
                nums.pop(index)
                nums.append(0)
            else:
                index += 1








