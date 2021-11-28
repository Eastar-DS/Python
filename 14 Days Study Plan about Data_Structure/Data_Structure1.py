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
Day 4 Array : 566. Reshape the Matrix, 118. Pascal's Triangle
Day 5 Array : 36. Valid Sudoku, 74. Search a 2D Matrix
Day 6 String : 
    387. First Unique Character in a String, 383. Ransom Note, 242. Valid Anagram
Day 7 Linked List : 
    141. Linked List Cycle, 21. Merge Two Sorted Lists, 203. Remove Linked List Elements
Day 8 Linked List : 
    206. Reverse Linked List, 83. Remove Duplicates from Sorted List
Day 9 Stack / Queue : 
    20. Valid Parentheses, 232. Implement Queue using Stacks


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

    
#day3    
    def maxProfit(self, prices: List[int]) -> int:
        minprice = prices[0]
        profit = 0
        for price in prices[1:]:
            profit = max(profit, price - minprice)
            minprice = min(minprice, price)
        return profit
            
            
            
        
        """
        카데인알고리즘의 뺄샘 버전이였네!
        prices[3] - prices[0] = 
            (prices[3] - prices[2]) + (prices[2] -prices[1]) + (prices[1] - prices[0])            
        """
        # length = len(prices)
        # if(length < 2) :
        #     return 0
        # profit = prices[1] - prices[0]
        # temp = prices[1] - prices[0]
        # index = 2
        # while(index < length):
        """
            3가지경우.
            과거의 어떤지점이 가장크거나, 현재지점과 과거의어떤지점을뺀게 가장크거나, current가크거나.(profit이 -인경우)
            
            [2,1,2,1,0,1,2]일때 1을반환하네...
        """
        #     current = prices[index] - prices[index - 1]
        #     if(profit < 0):
        #         profit,temp = current, current
        #     else:
        #         profit = max(profit, temp + current)
        #         temp += current
        #     index += 1
        # if(profit < 0):
        #     return 0
        # return profit
    
    
#day4
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        import numpy as np
        try:
            return np.reshape(mat, (r, c)).tolist()
        except:
            return mat
        

    # def matrixReshape2(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
    #     flat = sum(nums, [])
    #     if len(flat) != r * c:
    #         return nums
    #     tuples = zip(*([iter(flat)] * c))
    #     return map(list, tuples)

        
    #     return nums if len(sum(nums, [])) != r * c else map(list, zip(*([iter(sum(nums, []))]*c)))

    
    # def matrixReshape3(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
    #     if r * c != len(nums) * len(nums[0]):
    #         return nums
    #     it = itertools.chain(*nums)
    #     return [list(itertools.islice(it, c)) for _ in xrange(r)]
    
    
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]]
        for i in range(1, numRows):
            res += [map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1])]
        return res[:numRows]
  
    
    def generate2(numRows):
        pascal = [[1]*(i+1) for i in range(numRows)]
        for i in range(numRows):
            for j in range(1,i):
                pascal[i][j] = pascal[i-1][j-1] + pascal[i-1][j]
        return pascal
    
    
#day5
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
        Output: true
        
        Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
        Output: false
        """
        if(target < matrix[0][0] or matrix[-1][-1] < target):
            return False
        
        l,r = 0, len(matrix)
        row = (l+r)//2
        while(l + 1 != r):
            if(target >= matrix[row][0]):
                l = row
            else:
                r = row
            row = (l+r)//2
            
        if(target in matrix[row]):
            return True
        
        return False























    
    
    
    
    
    
    
    
    
    
    
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



#day3
    "Memory Usage 100%"
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        Input: nums1 = [1,2,2,1], nums2 = [2,2]
        Output: [2,2]
        
        Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
        Output: [4,9]
        """
        nums3 = []
        if(len(nums1) < len(nums2)):
            for num in nums1:
                if(num in nums2):
                    nums3.append(num)
                    nums2.remove(num)
        else:
            for num in nums2:
                if(num in nums1):
                    nums3.append(num)
                    nums1.remove(num)
        return nums3


    "Time limit ㅋㅋ"
    # def maxProfit(self, prices: List[int]) -> int:
    #     length = len(prices)
    #     profits = [0] * length
    #     for i in range(length-1,0,-1):
    #         for j in range(i):
    #             maximum = max(prices[i] - prices[j],profits[i])
    #             if(maximum > 0):
    #                 profits[i] = maximum
    #     return max(profits)

    "Time limit ㅋㅋ"
    # def maxProfit(self, prices: List[int]) -> int:
    #     profit = 0
    #     for future in range(len(prices)-1, 0, -1):
    #         maximum = max([prices[future] - prices[current] for current in range(0,future)])
    #         if(maximum > profit):
    #             profit = maximum
        
    #     return profit

    def maxProfit(self, prices: List[int]) -> int:
        """
        Input: prices = [7,1,5,3,6,4]
        Output: 5
        """
        length = len(prices)
        
        #[1]처리
        if(length < 2):
            return 0
        
        minvalues = [prices[0]] + [0]*(length - 2)
        minvalue = prices[0]
        
        for i in range(length - 1):
            if(prices[i] < minvalue):
                minvalue = prices[i]
                minvalues[i] = minvalue
            else:
                minvalues[i] = minvalue
                
        profit = max([(prices[j+1] - minvalues[j]) for j in range(length - 1)])
        if(profit > 0):
            return profit
        else:
            return 0

    
#day4
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        """
        Input: mat = [[1,2],[3,4]], r = 1, c = 4
        Output: [[1,2,3,4]]
        
        Input: mat = [[1,2],[3,4]], r = 2, c = 4
        Output: [[1,2],[3,4]]
        """

        m , n = len(mat), len(mat[0])
        if(m * n != r * c):
            return mat
        
        temp_mat = []
        output = []
        #temp_mat : [1,2,3,4,...] 처럼 만들기
        for array in mat:
            temp_mat += array 
        
        temp_array = []
        for i in range(m*n):
            if(i%c == c-1):
                temp_array.append(temp_mat[i])
                output.append(temp_array)
                temp_array = []
            else:
                temp_array.append(temp_mat[i])
        
        return output
            


    def generate(self, numRows: int) -> List[List[int]]:
        """
        Input: numRows = 5
        Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
        """
        if(numRows == 1):
            return [[1]]
        if(numRows == 2):
            return[[1],[1,1]]
        
        output = [[1],[1,1]]
        def Pascal(post_pascal:List[int]) -> List[int]:
            next_pascal = [1]
            for i in range(len(post_pascal) - 1):
                next_pascal.append(post_pascal[i] + post_pascal[i+1])
            next_pascal.append(1)
            return next_pascal              
                        
        for i in range(numRows - 2):
            output.append(Pascal(output[-1]))
            
        return output
        

#day5
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        #1 row
        for row in board:
            temp = []
            for element in row:
                if(element != "."):
                    temp.append(element)
            if(len(temp) != len(set(temp))):
                return False        
            
        #2 col
        for i in range(9):
            temp = [row[i] for row in board if(row[i] != ".")]
            
            if(len(temp) != len(set(temp))):
                return False
        #3 square
        for i in range(3):
            for j in range(3):
                temp = [board[3*i+x][3*j+y] for x in range(3) for y in range(3) if(board[3*i+x][3*j+y] != ".")]
                
                if(len(temp) != len(set(temp))):
                    return False
        
        return True




    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
        Output: true
        
        Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
        Output: false
        """
        if(target < matrix[0][0] or matrix[-1][-1] < target):
            return False
        
        row = 0
        while(row < len(matrix) and target >= matrix[row][0]):
            row += 1
            
        if(target in matrix[row - 1]):
            return True
        
        return False














































        