import collections
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from typing import *

class Solution(object):
    def twoSum(nums, target):
        for i in range(len(nums)-1):
            for j in range(len(nums)-(i+1)):
                Sum = nums[i] + nums[i+j+1]
                if((target - Sum) == 0):
                    Output = [i,i+j+1]
                    return Output
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

    def reverse(x):
        string = str(x)
        size = len(string)
        answer = ''
        if(x>2**31-1 or x<-2**31):
            return 0
        elif(x > 0):
            for i in range(size):
                answer = answer + string[size-(i+1)]
            if(int(answer)>(2**31)-1 or int(answer)<-(2**31)):
                return 0
        elif(x < 0):
            answer = '-'
            for i in range(size-1):
                answer = answer + string[size-(i+1)]
            if(int(answer)>(2**31)-1 or int(answer)<-(2**31)):
                return 0
        elif(x == 0):
            return 0;
        return int(answer)
    
        """
        :type x: int
        :rtype: int
        """
    
    def isPalindrome(x):
        """
        :type x: int
        :rtype: bool
        """
        if(x > 2**31 - 1 or x < -2**31):
            return 0
        string = str(x)
        size = int((len(string)+1)/2)
        for i in range(size):
            if(string[i] != string[len(string)-(i+1)]):
                return False
        return True    

# class Solution(object):
#     def addTwoNumbers(l1, l2):
#         """
#         :type l1: ListNode
#         :type l2: ListNode
#         :rtype: ListNode
#         """
#         l3=[]
#         len1 = len(l1)
#         len2 = len(l2)
#         minlen = min(len1,len2)
#         carry1 = 0
#         for i in range(minlen):
#             if(l1[i] + l2[i] + carry1 > 9):
#                 l3.append(l1[i]+l2[i] - 10 + carry1)
#                 carry1 = 1
#             else:
#                 l3.append(l1[i]+l2[i] + carry1)
#                 carry1 = 0
#         if(len1 == len2):
#             if(carry1 == 0):
#                 return l3
#             if(carry1 == 1):
#                 l3.append(1)
#                 return l3
#         maxlen = max(len1,len2)
#         for j in range(maxlen - minlen):
#             if(len1 > len2):
#                 if((l1[minlen + j] + carry1) < 10):
#                     l3.append(l1[minlen + j] + carry1)
#                     carry1 = 0
#                 else:
#                     l3.append(0)
#                     carry1 = 1
#             else:
#                 if((l2[minlen + j] + carry1) < 10):
#                     l3.append(l2[minlen + j] + carry1)
#                     carry1 = 0
#                 else:
#                     l3.append(0)
#                     carry1 = 1
#         if(carry1 == 1):
#             l3.append(1)
#         return l3
        
#1200. Minimum Absolute Difference
class Solution1200:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        """
        Minimum of two abs diff
        Input: arr = [4,2,1,3]
        Output: [[1,2],[2,3],[3,4]]
        
        Input: arr = [1,3,6,10,15]
        Output: [[1,3]]
        
        Runtime: 400 ms, faster than 30.97%
        Memory Usage: 35.5 MB, less than 5.06%
        """
        arr.sort()
        output,min_diff = [], arr[1] - arr[0]
        for i in range(len(arr)-1):
            min_diff = min(min_diff, arr[i+1] - arr[i])
            output.append([arr[i], arr[i+1]])
        output = [ele for ele in output if (ele[1] - ele[0]) == min_diff]
        return output


    def minimumAbsDifference1(self, arr: List[int]) -> List[List[int]]:
        """
        이거보다 빠른사람은 뭐냐...?
        Runtime: 332 ms, faster than 80.42%
        Memory Usage: 28.2 MB, less than 60.76%
        """
        arr.sort()
        output,minimum = [[arr[0],arr[1]]],arr[1] - arr[0]
        for i in range(1,len(arr) - 1):
            if(arr[i+1] - arr[i] < minimum):
                minimum = arr[i+1] - arr[i]
                output = [[arr[i],arr[i+1]]]
            elif(arr[i+1] - arr[i] == minimum):
                output.append([arr[i],arr[i+1]])
        return output


    def minimumAbsDifference2(self, arr: List[int]) -> List[List[int]]:
        """
        이게 위에거보다 빠르네;;? if문때문에 그런가? 
        다시 제출하니까 더느림. 데이터가 많아지면 더느려지지않을까.
        Runtime: 324 ms, faster than 89.20%
        Memory Usage: 28.3 MB, less than 24.57%
        """
        arr.sort()
        minimum = min(ele2-ele1 for ele1,ele2 in zip(arr,arr[1:]))
        return [[ele1,ele2] for ele1,ele2 in zip(arr,arr[1:]) if ele2-ele1 == minimum]
        




#231. Power of Two
class Solution231:
    def isPowerOfTwo(self, n: int) -> bool:
        """
        Runtime: 28 ms, faster than 88.11%
        Memory Usage: 14 MB, less than 97.84%
        """
        if(n<0):
            return False
        while(n>1):
            n /= 2
            if(n%2 > 1.0):
                return False
        if(n==1):
            return True
        return False


    def isPowerOfTwo1(self, n: int) -> bool:
        """
        Wow....
        16 = 10000 -> 15 = 01111 -> 16 & 15 = 00000!
        """
        return n > 0 and not (n & n-1)



#997. Find the Town Judge
class Solution997:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        """
        Input: n = 2, trust = [[1,2]]
        Output: 2
        
        Input: n = 3, trust = [[1,3],[2,3]]
        Output: 3
        
        Input: n = 3, trust = [[1,3],[2,3],[3,1]]
        Output: -1
        
        Runtime: 720 ms, faster than 83.59%
        Memory Usage: 18.8 MB, less than 97.64%
        """
        dic,li = collections.defaultdict(int),[1]*n
        for t in trust:
            li[t[0]-1],dic[t[1]] = 0, dic[t[1]] + 1
        for i in range(1,n+1):
            if(dic[i] == n-1):
                if(li[i-1] == 1):
                    return i
        return -1

#1009. Complement of Base 10 Integer
class Solution1009:
    def bitwiseComplement(self, n: int) -> int:
        """
        Input: n = 5
        Output: 2
        Explanation: 5 is "101" in binary, with complement "010" in binary, which is 2 in base-10.
        
        input = 0
        output = 1
        
        Runtime: 28 ms, faster than 80.61%
        Memory Usage: 14.1 MB, less than 89.54%
        """
        if(n==0):
            return 1
        for i in range(1,31):
            if(2**i>n):
                return ((2**i)-1)^n



#67. Add Binary
class Solution67:
    def addBinary(self, a: str, b: str) -> str:
        """
        Runtime: 45 ms, faster than 23.05%
        Memory Usage: 14.2 MB, less than 55.41%
        """
        output, s, c = "",0,0
        #padding
        if(len(a)>len(b)):
            b = '0'*(len(a)-len(b)) + b
        else:
            a = '0'*(len(b)-len(a)) + a
            
        for num1,num2 in zip(a[::-1],b[::-1]):
            s,c = (int(num1)+int(num2)+c)%2, (int(num1)+int(num2)+c)//2
            output = f'{s}' + output
        
        if(c==1):
            output = '1'+output
        
        return output








































