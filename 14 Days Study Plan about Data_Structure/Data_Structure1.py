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

Day 10 Tree : 
    144. Binary Tree Preorder Traversal, 94. Binary Tree Inorder Traversal, 145. Binary Tree Postorder Traversal
Day 11 Tree : 
    102. Binary Tree Level Order Traversal, 104. Maximum Depth of Binary Tree, 101. Symmetric Tree
Day 12 Tree : 
    226. Invert Binary Tree, 112. Path Sum
Day 13 Tree : 
    700. Search in a Binary Search Tree, 701. Insert into a Binary Search Tree
Day 14 Tree : 
    98. Validate Binary Search Tree, 653. Two Sum IV - Input is a BST, 235. Lowest Common Ancestor of a Binary Search Tree
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



#Day 6 String : 
#     387. First Unique Character in a String, 383. Ransom Note, 242. Valid Anagram
    def canConstruct(self, ransomNote, magazine):
        return not collections.Counter(ransomNote) - collections.Counter(magazine)


# Day 7 Linked List : 
#     141. Linked List Cycle, 21. Merge Two Sorted Lists, 203. Remove Linked List Elements
    def hasCycle(self, head):
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
        return False


    "try를 사용하는 풀이도 남겨두고싶어서 하나더적음"
    def hasCycle2(self, head):
        "48ms"
        try:
            slow = head
            fast = head.next
            while slow is not fast:
                slow = slow.next
                fast = fast.next.next
            return True
        except:
            return False
        
#디스커스에 있는 천재들아 고마워!        
    def mergeTwoLists(self, a, b):
        if a and b:
            if a.val > b.val:
                a, b = b, a
            a.next = self.mergeTwoLists(a.next, b)
        return a or b          
            
    def mergeTwoLists2(self, a, b):
        if not a or b and a.val > b.val:
            a, b = b, a
        if a:
            a.next = self.mergeTwoLists(a.next, b)
        return a



#Day 8 Linked List : 
    #206. Reverse Linked List, 83. Remove Duplicates from Sorted List
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        헤드를 다음으로 넘기고 현재 화살표를 이전꺼로 바꾸는 간단한아이디어
        Runtime: 32 ms, faster than 89.54%
        Memory Usage: 15.5 MB, less than 93.80%
        """
        prev = None
        while head:
            curr = head
            head = head.next
            curr.next = prev
            prev = curr
        return prev


    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        나는 벨류가 같으면 쭉 다음으로 넘겼는데 디스커스에서는 그다음껄로 가버림.        
        속도랑 메모리는 같았음.
        """
        if head is None:
            return head
        curr = head
        while curr.next:
            if curr.next.val == curr.val:
                curr.next = curr.next.next
            else:
                curr = curr.next
        return head















































































    
    
    
    
    
    
    
    
    
    
    
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

    
#Day 4 Array : 566. Reshape the Matrix, 118. Pascal's Triangle
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
        

#Day 5 Array : 36. Valid Sudoku, 74. Search a 2D Matrix
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



#Day 6 String : 
#     387. First Unique Character in a String, 383. Ransom Note, 242. Valid Anagram
    def firstUniqChar(self, s: str) -> int:
        """
        Input: s = "leetcode"
        Output: 0
        
        Input: s = "loveleetcode"
        Output: 2
        
        Input: s = "aabb"
        Output: -1
        """
        from collections import Counter
        count = Counter(s)
        for index, string in enumerate(s):
            if(count[string] == 1):
                return index
        return -1


    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        """
        Runtime: 44 ms
        
        Input: ransomNote = "a", magazine = "b"
        Output: false
        
        Input: ransomNote = "aa", magazine = "ab"
        Output: false
        
        Input: ransomNote = "aa", magazine = "aab"
        Output: true
        """
        from collections import Counter
        ran_count = Counter(ransomNote)
        mag_count = Counter(magazine)
        output = True
        for string in ran_count:
            count = ran_count[string]
            if(string in mag_count):
                if(mag_count[string] < count):
                    output = False
            else:
                output = False
        return output
        
#살짝 개선        
    def canConstruct2(self, ransomNote: str, magazine: str) -> bool:
        "Runtime: 40 ms, faster than 93.17%"
        from collections import Counter
        ran_count = Counter(ransomNote)
        output = True
        
        for string in ran_count:
            count = ran_count[string]
            if(string in magazine):
                if(magazine.count(string) < count):
                    output = False
            else:
                output = False
        return output


    def isAnagram(self, s: str, t: str) -> bool:
        from collections import Counter
        return Counter(s) == Counter(t)

    
# Day 7 Linked List : 
#     141. Linked List Cycle, 21. Merge Two Sorted Lists, 203. Remove Linked List Elements

    def hasCycle(self, head: Optional[ListNode]) -> bool:        
        #Definition for singly-linked list.
        class ListNode:
            def __init__(self, x):
                self.val = x
                self.next = None
        """
        Input: head = [3,2,0,-4], pos = 1
        Output: true
        
        Input: head = [1], pos = -1
        Output: false
        
        Runtime: 56 ms, faster than 65.77%
        Memory Usage: 18 MB, less than 17.48%
        """
        start = ListNode(None)
        start.next = head
        pos = -1
        index = 0
        dic = {}
        while(start.next != None):
            if(start.next not in dic):
                dic[start.next] = index
                start = start.next
                index += 1
            else:
                pos = dic[start.next]
                return True
        return False
                
        

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:        
        #Definition for singly-linked list.
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next
        """
        Input: list1 = [1,2,4], list2 = [1,3,4]
        Output: [1,1,2,3,4,4]
        
        Input: list1 = [], list2 = []
        Output: []
        
        Input: list1 = [], list2 = [0]
        Output: [0]
        
        Runtime: 40 ms, faster than 57.29%
        Memory Usage: 14.3 MB, less than 32.85%
        """
        head1, head2, output = ListNode(1,list1), ListNode(1,list2), ListNode()
        output_head = ListNode(None, output)
        
        while(head1.next != None):
            if(head2.next == None):
                output.next = head1.next
                return output_head.next.next
                
            if(head1.next.val <= head2.next.val):
                output.next = ListNode(head1.next.val)
                head1 = head1.next
                output = output.next
            else:
                output.next = ListNode(head2.next.val)
                head2 = head2.next
                output = output.next
        
        output.next = head2.next
        
        return output_head.next.next


    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        # Definition for singly-linked list.
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next
        """
        Input: head = [1,2,6,3,4,5,6], val = 6
        Output: [1,2,3,4,5]
        
        Input: head = [], val = 1
        Output: []
        
        Input: head = [7,7,7,7], val = 7
        Output: []
        
        Runtime: Runtime: 68 ms, faster than 73.71%
        Memory Usage: 18.8 MB, less than 8.90%
        """
        output = ListNode()
        output_head = ListNode(0,output)
        while(head):
            if(head.val != val):
                output.next = ListNode(head.val)
                output = output.next
                head = head.next
            else:
                head = head.next
                
        return output_head.next.next
        
        
#Day 8 Linked List : 
    #206. Reverse Linked List, 83. Remove Duplicates from Sorted List
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next
        """
        Input: head = [1,2,3,4,5]
        Output: [5,4,3,2,1]
        
        Input: head = [1,2]
        Output: [2,1]
        
        Input: head = []
        Output: []
        
        Runtime: 36 ms, faster than 72.74%
        Memory Usage: 20.2 MB, less than 5.60%
        """
        def reverse(past, curr, output = None):
            if(curr.next):
                output = reverse(past.next, curr.next, output)
            else:
                output = curr
            curr.next = past
            return output
        
        if(not head):
            return None
        if(not head.next):
            return head
        past = head
        curr = head.next
        output = reverse(past,curr)
        head.next = None
        
        return output
        
        
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next
        """
        Input: head = [1,1,2]
        Output: [1,2]
        
        Input: head = [1,1,2,3,3]
        Output: [1,2,3]
        
        Runtime: 40 ms, faster than 83.74%
        Memory Usage: 14.3 MB, less than 58.40%
        """
        if(not head):
            return head
        
        dummy1, dummy2 = head, head
        val = head.val
        while(dummy1.next):
            if(dummy1.next.val == val):
                dummy1 = dummy1.next
            else:
                dummy2.next = dummy1.next
                val = dummy1.next.val
                dummy1 = dummy1.next
                dummy2 = dummy2.next
        #마지막 중복숫자처리
        dummy2.next = None
        return head


# Day 9 Stack / Queue : 
#     20. Valid Parentheses, 232. Implement Queue using Stacks
    def isValid(self, s: str) -> bool:
        """
        Runtime: 32 ms, faster than 69.56%
        Memory Usage: 14.1 MB, less than 96.64%
        """
        valid_dic = {')':'(', '}':'{', ']':'['}
        first = ['(','{','[']
        stack = []
        for string in s:
            if(string in first):
                stack.append(string)
            else:
                if(not stack):
                    return False
                if(stack[-1] == valid_dic[string]):
                    stack.pop()
                else:
                    return False
        if(stack):
            return False
        return True
                
#232. Implement Queue using Stacks
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.inStack, self.outStack = [], []

    def push(self, x: int) -> None:
        self.inStack.append(x)

    def pop(self) -> int:
        self.move()
        return self.outStack.pop()

    def peek(self) -> int:
        self.move()
        return self.outStack[-1]

    def empty(self) -> bool:
        return (not self.inStack) and (not self.outStack) 


    def move(self):
        """
        :rtype nothing
        """
        if not self.outStack:
            while self.inStack:
                self.outStack.append(self.inStack.pop())
# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()



# Day 10 Tree : 
#     144. Binary Tree Preorder Traversal, 94. Binary Tree Inorder Traversal, 145. Binary Tree Postorder Traversal
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right
        """
        
        """
        



















        