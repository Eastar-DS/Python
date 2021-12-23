import collections
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from typing import *

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def addTwoNumbers(l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode

        Runtime: 52 ms, faster than 93.51% 
            of Python online submissions for Add Two Numbers.
        Memory Usage: 13.4 MB, less than 92.96% 
            of Python online submissions for Add Two Numbers.

        """
        l3 = None
        carry = 0
        node1 = l1
        node2 = l2
        val = node1.val + node2.val
        
        if(val + carry > 9):
                carry = 1
                val -= 10
        else:
                carry = 0
                
        l3 = ListNode(val)
        node3 = l3
        
        node1 = node1.next
        node2 = node2.next        
        while(node1 != None and node2 != None):
            val = node1.val + node2.val + carry
            if(val > 9):
                carry = 1
                val -= 10
            else:
                carry = 0
            node3.next = ListNode(val)
            node3 = node3.next
            node1 = node1.next
            node2 = node2.next
        
        if(node2 == None):
            node12 = node1
        else:
            node12 = node2    
        
        while(node12 != None):
            val = node12.val + carry
            if(val > 9):
                carry = 1
                val -= 10
            else:
                carry = 0
            node3.next = ListNode(val)
            node3 = node3.next
            node12 = node12.next       
        
        if(carry == 1):
            node3.next = ListNode(1)
            carry = 0
        
        return l3
           
         




    # def threeSum(nums):        
            # """
            # :type nums: List[int]
            # :rtype: List[List[int]]
            # [0,0,0,0], [-1,0,1,2,-1,-4,-2,-3,3,0,4]
            # [13,4,-6,-7,-15,-1,0,-1,0,-12,-12,9,3,-14,-2,-5,-6,7,8,2,-4,6,-5,-10,-4,-9,-14,-14,12,-13,-7,3,7,2,11,7,9,-4,13,-6,-1,-14,-12,9,9,-6,-11,10,-14,13,-2,-11,-4,8,-6,0,7,-12,1,4,12,9,14,-4,-3,11,10,-9,-8,8,0,-1,1,3,-15,-12,4,12,13,6,10,-4,10,13,12,12,-2,4,7,7,-15,-4,1,-15,8,5,3,3,11,2,-11,-12,-14,5,-1,9,0,-12,6,-1,1,1,2,-3]
            # Time Limit Exceeded...
            # """
            # def IsSame(l1,l2):
            #     l22 = l2.copy()
            #     for x in l1:
            #         for y in l22:
            #             if(x == y):                            
            #                 l22.remove(y)
            #                 break
                    
            #     if(l22 == []):
            #         return True
            #     else:
            #         return False
                
            # out = []
            # size = len(nums)
            # if(size < 3):
            #     return out
            
            # for i in range(size-2):
            #     for j in range(size-2-i):
            #         for k in range(size-2-(i+j)):
            #             val = nums[i]+nums[i+j+1]+nums[i+j+k+2]
            #             if(val == 0):
            #                 out.append([nums[i],nums[i+j+1],nums[i+j+k+2]])            
            
            # print(out)
            
            # for a in range(len(out) - 1):
            #     print(a)
            #     for b in range(len(out) - 1, a, -1):
            #         if(len(out) != a+1):
            #             if(IsSame(out[a],out[b])):
            #                 print("Complete")
            #                 del out[b]
            #                 print(out)
                
            # return out
            
        
    def threeSum(nums):
    #허허허...
    # '''
    # Runtime: 900 ms, faster than 53.66% of Python3 online submissions for 3Sum.
    # Memory Usage: 18.2 MB, less than 12.95% of Python3 online submissions for 3Sum.
    # '''
        res = list()
        nums = sorted(nums)

        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left = i + 1
            right = len(nums) - 1

            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum < 0:
                    left += 1
                elif sum > 0:
                    right -= 1
                else:
                    res.append(sorted([nums[i], nums[left], nums[right]]))
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1

        return res




    def threeSumClosest(nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        Time Limit Exceeded
        [36,38,95,-89,-86,-19,63,-8,12,90,15,-84,48,50,88,88,-29,-2,99,-97,60,88,30,64,-28,-87,2,78,87,97,77,63,77,62,89,57,39,-36,39,-43,86,76,32,-71,-46,58,18,-27,52,-68,-79,-54,0,18,-88,72,-57,95,-66,73,-99,33,-16,43,81,40,0,-8,-15,6,87,-43,92,-64,68,1,-32,15,-60,-49,35,31,49,-70,65,0,-87,27,12,2,-94,79,4,41,19,-37,-79,-22,7,-25,-67,-56,34,-64,-7,-58,2,26,98,2,23,2,7,62,49,-18,44,-1,91,56,64,-98,-84,38,23,63,-80,14,56,-100,-62,19,24,-16,18,-78,-52,47,99,82,-91,-34,76,89,-56,-35,-72,-90,41,43,-43,6,-95,-63,-70,-81,-55,-63,-28,-61,-72,68,-50,72,-28,83,67,99,41,54,73,-4,14,-91,51,93,46,32,-49,87,-84,-13,57,12,74,42,33,39,-79,-56,-46,-53,-74,-88,55,-65,-75,-89,-56,97,100,7,84,79,8,24,48,-46,-95,76,73,-87,85,45,-8,-69]
171
        Runtime: 88 ms, faster than 85.39% of Python online submissions for 3Sum Closest.
        Memory Usage: 13.4 MB, less than 77.30% of Python online submissions for 3Sum Closest.
        """
        diff = float('inf')
        out = 0
        nums = sorted(nums)
        for i in range(len(nums) - 2): 
            l = i+1
            r = len(nums) - 1
            while(l < r):
                Sum = nums[i] + nums[l] + nums[r] - target
                if(abs(Sum) < diff) : 
                    diff = abs(Sum)
                    out = nums[i] + nums[l] + nums[r]
                if(Sum == 0):
                    return out
                if(Sum < 0):
                    l += 1
                else:
                    r -= 1
        return out
            
        # for i in range(len(nums) - 2):
        #     for j in range(len(nums) - (i + 2)):
        #         for k in range(len(nums) - (i+j+2)):
        #             if(abs(nums[i] + nums[i+j+1] + nums[i+j+k+2] - target) < diff):
        #                 diff = abs(nums[i] + nums[i+j+1] + nums[i+j+k+2] - target)
        #                 out = (nums[i] + nums[i+j+1] + nums[i+j+k+2])
        #                 # print(nums[i], nums[i+j+1], nums[i+j+k+2], diff)
        #                 if(diff == 0) :
        #                     return out
        
        return out


    def letterCombinations(digits):
        """
        :type digits: str
        :rtype: List[str]
        0 <= digits.length <= 4
        8 ms	13.5 MB
        """
        # digitslist = []
        # out = []
        # dic = {'2':'abc', '3':'def', '4':'ghi', '5':'jkl', '6':'mno', '7':'pqrs', '8':'tuv', '9':'wxyz'}
        # length = len(digits)
        # for i in range(length):
        #     digitslist.append(digits[i])
        
            
        # X = digitslist[0]
        # if(digitslist[1] != None):
        #     Y = digitslist[1]
        # if(digitslist[2] != None):
        #     Z = digitslist[2]
        # if(digitslist[3] != None):
        #     W = digitslist[3]
        from functools import reduce
        if '' == digits: return []
        dic = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        return reduce(lambda z,w: [x+y for x in z for y in dic[w]], digits, [''])   





    # return reduce(lambda z,w: [x+y for x in z for y in dic[w]], digits, [''])




# reduce(lambda x,y: x+y, [1,2,3,4,5], 100)


# reduce(lambda z,w: [x+y for x in z for y in w], ['abc','def'], [''])







# reduce(lambda z,w: [x+y for x in z for y in dic[w]], digits, [''])



# reduce(lambda z,w: [x+y for x in z for y in w], ['abc'], [''])
# ['a', 'b', 'c']

# for x in ['']:
#    for y in ['abc']:
#       [x+y]
      


    def fourSum(nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        0 <= nums.length <= 200
        -10^9 <= nums[i] <= 10^9
        -10^9 <= target <= 10^9
        특이점들.
        [0,0,0,0],0
        [-1,-5,-5,-3,2,5,0,4],-7
        Runtime: 1620 ms, faster than 6.15% of Python online submissions for 4Sum.
        Memory Usage: 13.2 MB, less than 97.27% of Python online submissions for 4Sum. 
        """
       
        if(len(nums) < 4):
            return []
        nums = sorted(nums)
        length = len(nums)
        res = []
        i = 0
        while( i < (length - 3)):
            target2 = target - nums[i]
            idx = i + 1
            while( idx < (length-2)):                
                l = idx + 1
                r = length - 1
               
                while( l < r ):
                    if((nums[idx] + nums[l] + nums [r]) == target2):
                        res.append([nums[i], nums[idx], nums[l], nums[r]])
                        # print('append', i, idx, l, r)
                        while((l < length - 1) and (nums[l] == nums[l+1])):
                            l += 1
                        l += 1
                        continue
                    if((nums[idx] + nums[l] + nums [r]) < target2):                        
                        while((l < length - 1) and (nums[l] == nums[l+1])):
                            l += 1
                        l += 1
                        continue
                    if((nums[idx] + nums[l] + nums [r]) > target2):                        
                        while((r > idx + 1) and (nums[r] == nums[r-1])):
                            r -= 1
                        r -= 1
                        continue
                
                while((idx < length - 2) and (nums[idx] == nums[idx + 1])):
                    idx += 1
                idx += 1
            while( (i < length - 3) and (nums[i] == nums[i+1])):
                i += 1
            i += 1    
        return res
        '''
        while((l < length - 1) and (nums[l] == nums[l+1]))의 앞의 조건은 l이 계속해서 커지다가 리스트의 끝까지 가버렸을때를
        대비하기위함.
        '''                    

#394. Decode String
class Solution394:
    def decodeString(self, s: str) -> str:
        """
        Input: s = "3[a]2[bc]"
        Output: "aaabcbc"
        
        Input: s = "3[a2[c]]"
        Output: "accaccacc"
        
        Input: s = "abc3[cd]xyz"
        Output: "abccdcdcdxyz"
        
        "3[z]2[2[y]pq4[2[jk]e1[f]]]ef" 숫자뒤에 숫자가 또올 수 있을줄 생각 못했다.
        
        """

        i, stack, output = 0, [], ""
        start, end, temp = 0, 0, ""
        #making stack
        while(i < len(s)):
            if(s[i].isalnum()):
                temp += s[i]
            elif(s[i] == '['):
                start += 1
            elif(s[i] == ']'):
                end += 1
            
            if(start != 0 and start == end):
                stack.append(temp)
                start, end, temp = 0, 0, ""
                
            i+= 1
        if(temp):
            stack.append(temp)
        #making output
        while(stack):
            pop = stack.pop()
            j, alpha, num = len(pop)-1,"",""
            while(j>-1):
                if(pop[j].isalpha()):
                    alpha = pop[j] + alpha
                else:
                    num = pop[j] + num
                    if (j-1 > -1 and pop[j-1].isdigit()):
                        j -= 1                        
                        continue                        
                    else:
                        alpha = alpha * int(num)
                        num = ""
                j -= 1
                                            
            output = alpha + output
            alpha = ""
        
        return output



    def decodeString1(self, s: str) -> str:
        """
        Input: s = "3[a2[c]]"
        Output: "accaccacc"
        
        Input: s = "abc3[cd]xyz"
        Output: "abccdcdcdxyz"
        
        "3[z]2[2[y]pq4[2[jk]e1[f]]]ef"
        
        Runtime: 28 ms, faster than 83.89%
        Memory Usage: 14.3 MB, less than 19.69%
        """
        stack = []; curNum = 0; curString = ''
        for c in s:
            #처음에 숫자로시작해도 curString = '' 을 스택에 추가
            if c == '[':
                stack.append(curString)
                stack.append(curNum)
                curString = ''
                curNum = 0
            #이걸 어케알았누? 대단해...
            elif c == ']':
                num = stack.pop()
                prevString = stack.pop()
                curString = prevString + num*curString
            #이 아이디어 좋다!
            elif c.isdigit():
                curNum = curNum*10 + int(c)
            else:
                curString += c
        return curString




#143. Reorder List
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution143:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        Runtime: 84 ms, faster than 94.16%
        Memory Usage: 23.4 MB, less than 47.53%
        
        Input: head = [1,2,3,4]
        Output: [1,4,2,3]
        
        Input: head = [1,2,3,4,5]
        Output: [1,5,2,4,3]
        """
        dummy1, head_list  = head, []
        dummy2= head
        while(dummy1.next):
            dummy1 = dummy1.next
            head_list.append(dummy1)
        
        length = len(head_list)
        for i in range(length//2):
            dummy2.next = head_list[length - 1 - i]
            dummy2 = dummy2.next
            dummy2.next = head_list[i]
            dummy2 = dummy2.next
        if(length % 2 == 1):
            dummy2.next = head_list[length//2]
            dummy2 = dummy2.next
        dummy2.next = None
            
        

    def reorderList1(self, head):
        #step 1: find middle
        if not head: return []
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        #step 2: reverse second half
        """
        The idea is to keep three pointers: prev, curr, nextt stand for previous, 
            current and next and change connections in place. 
        Do not forget to use slow.next = None, in opposite case you will have list with loop.
        """
        prev, curr = None, slow.next
        while curr:
            nextt = curr.next
            curr.next = prev
            prev = curr
            curr = nextt    
        slow.next = None
        
        #step 3: merge lists
        
        
        head1, head2 = head, prev
        while head2:
            nextt = head1.next
            head1.next = head2
            head1 = head2
            head2 = nextt


#207. Course Schedule
class Solution207:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Input: numCourses = 2, prerequisites = [[1,0]]
        Output: true
        
        Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
        Output: false
        """
        dic = collections.defaultdict(list)
        for prerequisite in prerequisites:
            dic[prerequisite[0]].append(prerequisite[1])
        
        
        
        def dfs():
            




#210. Course Schedule II
class Solution210:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """
        Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
        Output: [0,2,1,3]
        
        Input: numCourses = 1, prerequisites = []
        Output: [0]
        
        그래프 이수업을 듣기위해 들어야하는 수업들
        """
        output = []
        dic = collections.defaultdict(list)
        for prerequisite in prerequisites:
            dic[prerequisite[0]].append(prerequisite[1])
        
        def DFS(key,connect):
            if(len(dic[key]) == 0):
                output.append(key)
                for values in dic.values():
                    if(key in values):
                        values.remove(key)
                return
            
            for value in dic[key]:
                if(value in connect):
                    return []
                DFS(value,connect.append(value))
        
        for key in dic.keys():
            connect = [key]
            DFS(key,connect)
                    
        for i in range(numCourses):
            if(i not in output):
                output.append(i)
        
        return output
























        