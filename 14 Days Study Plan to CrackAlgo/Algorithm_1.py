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
Day3 Two Pointers : 283. Move Zeroes, 167. Two Sum II - Input Array Is Sorted
Day4 Two Pointers : 344. Reverse String, 557. Reverse Words in a String III
Day5 Two Pointers : 876. Middle of the Linked List, 19. Remove Nth Node From End of List
Day 6 Sliding Window : 3. Longest Substring Without Repeatin, 567. Permutation in String
Day 7 Breadth-First Search / Depth-First Search : 
    733. Flood Fill, 695. Max Area of Island
Day 8 Breadth-First Search / Depth-First Search : 
    617. Merge Two Binary Trees, 116. Populating Next Right Pointers in Each Node
Day 9 Breadth-First Search / Depth-First Search : 
    542. 01 Matrix, 994. Rotting Oranges
 
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


#Day3
    def moveZeroes(self, nums: list) -> None:
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0 and nums[slow] == 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]

            # wait while we find a non-zero element to
            # swap with you
            if nums[slow] != 0:
                slow += 1


    #def twoSum(self, numbers: List[int], target: int) -> List[int]:
     

#day4
    def reverseString(self, s: List[str]) -> None:
        s[:] = s[::-1]


    def reverseWords(self, s):
        """
        output을 거꾸로해보면 띄어쓰기단위로 거꾸로 배열됨. 
        그렇다면 인풋을 띄어쓰기단위로 거꾸로만들고 나온 스트링을
        거꾸로하면 output이 나오겠구나!
        """
        return ' '.join(s.split()[::-1])[::-1]

    def reverseWords2(self, s):
        return ' '.join(x[::-1] for x in s.split())


    """
    >>> from timeit import timeit
    >>> setup = 's = "Let\'s take LeetCode contest"'
    >>> statements = ("' '.join(s.split()[::-1])[::-1]",
	          "' '.join(x[::-1] for x in s.split())",
	          "' '.join([x[::-1] for x in s.split()])")
    >>> for stmt in statements:
        print ' '.join('%.2f' % timeit(stmt, setup) for _ in range(5)), 'seconds for:', stmt

    0.79 0.78 0.80 0.82 0.79 seconds for: ' '.join(s.split()[::-1])[::-1]
    2.10 2.14 2.08 2.06 2.13 seconds for: ' '.join(x[::-1] for x in s.split())
    1.27 1.26 1.28 1.28 1.26 seconds for: ' '.join([x[::-1] for x in s.split()])
    
    
    >>> setup = 's = "Let\'s take LeetCode contest" * 1000'
    >>> for stmt in statements:
        print ' '.join('%.2f' % timeit(stmt, setup, number=1000) for _ in range(5)), 'seconds for:', stmt

    0.16 0.14 0.13 0.14 0.14 seconds for: ' '.join(s.split()[::-1])[::-1]
    0.69 0.71 0.69 0.70 0.70 seconds for: ' '.join(x[::-1] for x in s.split())
    0.63 0.68 0.63 0.64 0.64 seconds for: ' '.join([x[::-1] for x in s.split()])
    """



#day5
    def middleNode(self, head):
        "아이디어 진짜 심플하고 좋다..."
        tmp = head
        while tmp and tmp.next:
            head = head.next
            tmp = tmp.next.next
        
        return head



    def removeNthFromEnd(self, head, n):
        dummy = ListNode(0)
        dummy.next = head
        fast = slow = dummy
        for i in range(n):
            fast = fast.next
        while fast and fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next



























































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



#day3
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



    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        """
        Input: numbers = [2,7,11,15], target = 9
        Output: [1,2]
        
        Input: numbers = [-1,0], target = -1
        Output: [1,2]
        """
        dic = {}
        for idx, num in enumerate(numbers):
            if((target - num) in dic):
                return[dic[target - num] + 1, idx + 1]
                
            else:
                #idx1
                dic[num] = idx


#day4
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        You must do this by modifying the input array in-place with O(1) extra memory.
        
        Input: s = ["h","e","l","l","o"]
        Output: ["o","l","l","e","h"]
        
        Input: s = ["H","a","n","n","a","h"]
        Output: ["h","a","n","n","a","H"]
        """
        left, right = 0, len(s) - 1
        while(left < right):
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1


    def reverseWords(self, s: str) -> str:
        """
        Input: s = "Let's take LeetCode contest"
        Output: "s'teL ekat edoCteeL tsetnoc"
        """
        output = ''
        word = ''
        for string in s:            
            if(string != ' '):
                word = string + word
            else:
                output = output + word + string
                word = ''
        
        if(word):
            output += word
            
        return output



#Day5 Two Pointers : 876. Middle of the Linked List, 19. Remove Nth Node From End of List
#Definition for singly-linked list.
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next
            
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Input: head = [1,2,3,4,5]
        Output: [3,4,5]
        
        Input: head = [1,2,3,4,5,6]
        Output: [4,5,6]
        """
        length = 1
        len_head = head
        while(len_head.next):
            length += 1
            len_head = len_head.next
        
        length //= 2
        
        output = head
        while(length):
            output = output.next
            length -= 1

        return output


    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        Input: head = [1,2,3,4,5], n = 2
        Output: [1,2,3,5]
        
        Input: head = [1], n = 1
        Output: []
        
        Input: head = [1,2], n = 1
        Output: [1]
        
        """
        if(not head.next):
            return head.next
            
        length = 1
        len_head = head
        while(len_head.next):
            length += 1
            len_head = len_head.next
            
        #[1,2], n = 2 오류수정
        if(length == n):
            return head.next
        
        index = length - n - 1
        head2 = head
        while(index):
            head2 = head2.next
            index -= 1        
            
        head2.next = head2.next.next
        
        return head


#Day 6 Sliding Window : 3. Longest Substring Without Repeatin, 567. Permutation in String
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        이문제때문에 하루가 날아갔네 ㅡㅡ;
        Input: s = "abcabcbb"
        Output: 3
        
        Input: s = "abba"
        Output: 2
        
        Input: s = " "
        Output: 1
        
        Input: s = ""
        Output: 0
        """
        dic_string = {}
        length = 0
        output = 0
        for index,string in enumerate(s):
            if(string in dic_string and dic_string[string] >= index - length):                
                length = index - dic_string[string]
            else:                
                length += 1
                # print("e", index, past, current)
            dic_string[string] = index
            output = max(length, output)
            
        return output


















# Day 7 Breadth-First Search / Depth-First Search : 
#     733. Flood Fill, 695. Max Area of Island




























































