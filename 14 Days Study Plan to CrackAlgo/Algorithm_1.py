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
    
Day 10 Recursion / Backtracking : 21. Merge Two Sorted Lists, 206. Reverse Linked List
Day 11 Recursion / Backtracking : 
    77. Combinations, 46. Permutations, 784. Letter Case Permutation
Day 12 Dynamic Programming : 
    70. Climbing Stairs, 198. House Robber, 120. Triangle
Day 13 Bit Manipulation : 231. Power of Two, 191. Number of 1 Bits
Day 14 Bit Manipulation : 190. Reverse Bits, 136. Single Number

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

#Day 8 Breadth-First Search / Depth-First Search : 
    # 617. Merge Two Binary Trees, 116. Populating Next Right Pointers in Each Node
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        "내풀이를 훨신 짧게 적어놨네 속도는 더느림."
        if not root1 and not root2: return None
        ans = TreeNode((root1.val if root1 else 0) + (root2.val if root2 else 0))
        ans.left = self.mergeTrees(root1 and root1.left, root2 and root2.left)
        ans.right = self.mergeTrees(root1 and root1.right, root2 and root2.right)
        return ans


    def connect1(self, root):
        if root and root.left and root.right:
            root.left.next = root.right
            if root.next:
                root.right.next = root.next.left
        self.connect(root.left)
        self.connect(root.right)
 
    # BFS       
    def connect2(self, root):
        if not root:
            return 
        queue = [root]
        while queue:
            curr = queue.pop(0)
            if curr.left and curr.right:
                curr.left.next = curr.right
                if curr.next:
                    curr.right.next = curr.next.left
            queue.append(curr.left)
            queue.append(curr.right)
    
    # DFS 
    def connect3(self, root):
        if not root:
            return 
        stack = [root]
        while stack:
            curr = stack.pop()
            if curr.left and curr.right:
                curr.left.next = curr.right
                if curr.next:
                    curr.right.next = curr.next.left
            stack.append(curr.right)
            stack.append(curr.left)


#Day 11 Recursion / Backtracking : 
    # 77. Combinations, 46. Permutations, 784. Letter Case Permutation
    
#recursive
    def combine1(self, n: int, k: int) -> List[List[int]]:
        if k == 0:
            return [[]]
        return [pre + [i] for i in range(k, n+1) for pre in self.combine1(i-1, k-1)]

#iterative
    def combine2(self, n, k):
        combs = [[]]
        for _ in range(k):
            combs = [[i] + c for c in combs for i in range(1, c[0] if c else n+1)]
        return combs
#Reduce
    def combine3(self, n, k):
        from functools import reduce
        return reduce(lambda C, _: [[i]+c for c in C for i in range(1, c[0] if c else n+1)],
                  range(k), [[]])








































































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
            dic_string[string] = index
            output = max(length, output)
            
        return output


    def checkInclusion(self, s1: str, s2: str) -> bool:
        """
        Input: s1 = "ab", s2 = "eidbaooo"
        Output: true
        
        Input: s1 = "ab", s2 = "eidboaoo"
        Output: false
        """
        from collections import Counter
        len1 = len(s1)
        len2 = len(s2)
        if(len2 < len1):
            return False
        
        #permutation 만들지말고, 안의 구성요소가 같은지 판단하면됨
        #카운터라는 엄청나게 좋은 기능이있다.        
        tool = Counter(s1)
        for start in range(len2 - len1 + 1):
            if(tool == Counter(s2[start:len1+start])):
                return True
            
        return False

# Day 7 Breadth-First Search / Depth-First Search : 
#     733. Flood Fill, 695. Max Area of Island
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        """
        한점에서 시작해서 사방으로 벨류를 바꾸는문제
        """
        m = len(image)
        n = len(image[0])
        currentColor = image[sr][sc]
        
        def ChangeColor(image, i,j):
            image[i][j] = newColor
            
            if(i>0 and image[i-1][j] == currentColor):
                ChangeColor(image, i-1, j)
                
            if(i<m-1 and image[i+1][j] == currentColor):
                ChangeColor(image, i+1, j)
                
            if(j>0 and image[i][j-1] == currentColor):
                ChangeColor(image, i, j-1)
                
            if(j<n-1 and image[i][j+1] == currentColor):
                ChangeColor(image, i, j+1)
                
        if(currentColor != newColor):
            ChangeColor(image, sr, sc)
            
        return image
        

#속도 99.97퍼!
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        """
        물은 0, 땅은 1의 벨류를 갖는다. 가장 큰 땅을 구해라.
        """
        m,n = len(grid), len(grid[0])
        output = 0
        grid2 = grid.copy()
        def LandSize(i,j,size = 0) -> int:
            size += 1
            grid2[i][j] = 0                        
          
            if(i < m-1 and grid2[i+1][j] == 1):
                size = LandSize(i+1,j,size)
                    
            if(j < n-1 and grid2[i][j+1] == 1):
                size = LandSize(i,j+1,size)
            
            if(i > 0 and grid2[i-1][j] == 1):
                size = LandSize(i-1,j,size)
                
            if(j > 0 and grid2[i][j-1] == 1):
                size = LandSize(i,j-1,size)
                
            return size


        for i in range(m):
            for j in range(n):
                if(grid2[i][j] == 1):
                    output = max(output, LandSize(i,j))
                
        return output


#Day 8 Breadth-First Search / Depth-First Search : 
    # 617. Merge Two Binary Trees, 116. Populating Next Right Pointers in Each Node
    def mergeTrees(self, root1, root2):
        # Definition for a binary tree node.
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right
        """
        Note: The merging process must start from the root nodes of both trees.
        
        Input: root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
        Output: [3,4,5,5,4,null,7]
        
        Input: root1 = [1], root2 = [1,2]
        Output: [2,2]
        
        Runtime: 88 ms, faster than 63.31%
        Memory Usage: 15.6 MB, less than 27.97%
        """
        output = TreeNode()
        if(root1 == None and root2 == None):
            return None
        
        if(root1 != None and root2 == None):
            output = root1
        if(root1 == None and root2 != None):
            output = root2
        
        #root가 둘다 존재하는경우에만 함수를 다시부르면됨.
        if(root1 != None and root2 != None):
            output.val = root1.val + root2.val
            if((root1.left != None) or (root2.left != None)):
                output.left = self.mergeTrees(root1.left, root2.left)
            if((root1.right != None) or (root2.right != None)):
                output.right = self.mergeTrees(root1.right, root2.right)        
             
        return output
        
        

    def connect(self, root: 'Node') -> 'Node':
        class Node:
            def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
                self.val = val
                self.left = left
                self.right = right
                self.next = next
        """
        Runtime: 52 ms, faster than 98.41%
        Memory Usage: 15.6 MB, less than 92.03%
        """
        if(not root):
            return root
        save = root.left
        head = root
        #while 조건 신경써야
        while(head.left):
            head.left.next = head.right                
            if(head.next):
                head.right.next = head.next.left
                head = head.next
            else:
                head = save
                save = save.left
        
        return root


# Day 9 Breadth-First Search / Depth-First Search : 
#     542. 01 Matrix, 994. Rotting Oranges

    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        """
        Input: mat = [[0,0,0],[0,1,0],[1,1,1]]
        Output: [[0,0,0],[0,1,0],[1,2,1]]
        
        Runtime: 1264 ms, faster than 7.95%
        Memory Usage: 33.3 MB, less than 6.41%
        하루종일 이것만했다 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ^
        덕분에 데크도 써봅니다....        
        """
        #output = [[-1]*(n+2)]*(m+2) 로 하니까 원소하나를 바꾸면 한열의 모든원소가 바뀜 ㅡㅡ
        m,n = len(mat), len(mat[0])
        import numpy as np
        output = np.array([[-1]*(n+2)]*(m+2))
        
        from collections import deque
        indexlist = deque([])
        
        for i in range(m):
            for j in range(n):
                if(mat[i][j] == 0):
                    output[i+1][j+1] = 0
                    indexlist.append([i+1,j+1])
                    
        
        # while(-1 in output[1:-1][k] for k in range(1,n+1)):
        while(indexlist):
            [i,j] = indexlist.popleft()
            dis = output[i][j] + 1
                
            index = [[i,j+1],[i+1,j],[i,j-1],[i-1,j]]
            
            for [x,y] in index:
                if(x <1 or y<1 or x>m or y>n):
                    continue
                
                if(output[x][y] == -1):
                    output[x][y] = dis
                    indexlist.append([x,y])
        
        #output 정리
        output = output[1:-1,1:-1]
            
        return output


    def orangesRotting(self, grid: List[List[int]]) -> int:
        """
        Runtime: 104 ms, faster than 5.33%
        Memory Usage: 31.1 MB, less than 10.89%
        
        넘모어렵다....
        """
        from collections import deque
        two_list = deque([])
        import numpy as np
        npgrid = np.array(grid)
        
        m,n = len(grid), len(grid[0])
        
        #2골라서 저장
        for i in range(m):
            for j in range(n):
                if(npgrid[i][j] == 2):
                    two_list.append([i,j,0])
        
        #2가 없을때 1이 있는지확인.
        if(not two_list):
            for i in range(m):
                for j in range(n):
                    if(npgrid[i][j] == 1):
                        return -1
            return 0
        
        #상하좌우 썩게만드는함수
        def MakeRotten(x,y,time):
            if(x>0 and grid[x-1][y] == 1):
                grid[x-1][y] = 2
                two_list.append([x-1,y,time+1])
            if(x<m-1 and grid[x+1][y] == 1):
                grid[x+1][y] = 2
                two_list.append([x+1,y,time+1])
            if(y>0 and grid[x][y-1] == 1):
                grid[x][y-1] = 2
                two_list.append([x,y-1,time+1])
            if(y<n-1 and grid[x][y+1] == 1):
                grid[x][y+1] = 2
                two_list.append([x,y+1,time+1])
            
        #다썩을때까지 걸리는시간구하기.        
        while(two_list):
            [x,y,time] = two_list.popleft()
            MakeRotten(x,y,time)
        
        #1있으면 -1반환
        for i in range(m):
            for j in range(n):
                if(grid[i][j] == 1):
                    return -1        
        
        return time
            

#Day 10 Recursion / Backtracking : 21. Merge Two Sorted Lists, 206. Reverse Linked List
    "다 푼거네."


#Day 11 Recursion / Backtracking : 
    # 77. Combinations, 46. Permutations, 784. Letter Case Permutation

    def combine(self, n: int, k: int) -> List[List[int]]:
        """
        [1,n]사이 k개숫자로 이루어진 리스트
        
        Input: n = 4, k = 2
        Output:
            [
                [2,4],
                [3,4],
                [2,3],
                [1,2],
                [1,3],
                [1,4],
            ]
            
        Input: n = 1, k = 1
        Output: [[1]]
        
        Runtime: 76 ms, faster than 97.46%
        Memory Usage: 15.6 MB, less than 81.04%
        """
        from itertools import combinations
        input_num = []
        for i in range(1,n+1):
            input_num.append(i)
        output = []
        c = combinations(input_num, k)
        for element in c:
            output.append(list(element))
        return output


    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        Input: nums = [1,2,3]
        Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
        
        Runtime: 36 ms, faster than 90.06%
        Memory Usage: 14.3 MB, less than 90.15%
        이건 할만했다 휴...
        """
        output = []
        if(len(nums) > 1):
            for num in nums:
                array = nums.copy()
                array.remove(num)
                output += [[num] + per for per in self.permute(array)]                
        else:
            return []
        
        return output

    def permute1(self, nums):
        return list(itertools.permutations(nums))

    
    def letterCasePermutation(self, s: str) -> List[str]:
        """
        디스커스 참고후품
        
        Runtime: 67 ms, faster than 36.30%
        Memory Usage: 14.7 MB, less than 92.14%
        """
        output = ['']
        for string in s:
            if(string.isalpha()):
                output = [a + b for a in output for b in [string, string.swapcase()]]
            else:
                output = [a + string for a in output]
        
        return output
                    
        













