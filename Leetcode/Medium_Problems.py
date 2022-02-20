import collections
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from typing import *

#211

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
        trace,visited = set(),set()        
        
        def dfs(i):
            if i in trace:
                return False
            if i in visited:
                return True
            
            trace.add(i)
            for value in dic[i]:
                if not dfs(value):
                    return False
            trace.remove(i)
            visited.add(i)
            return True
        
        #list(dic)으로 쓰는거 주의
        for key in list(dic):
            if not dfs(key):
                return False
        
        return True
        
    
    
    def canFinish1(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    	# 선수과목 관련 ref 테이블 먼저 만들어 줌
        # ref[i]는 i를 듣기 위해 먼저 들어야 하는
        # 과목들의 집합임!
        ref = [[] for _ in range(numCourses)]
        for x, y in prerequisites:
            ref[x].append(y)
        
        visited = [0] * numCourses
        
        # 각 노드 돌며 사이클 생성 확인
        for i in range(numCourses):
            if not self.dfs(ref, visited, i): #사이클생성
            	return False
        return True
        
    # 깊이우선탐색 함수
    def dfs(self, ref, visited, i):
    	# -1: 방문 중인 노드
        # 1: 이미 방문한 노드
        if visited[i] == 1:
            return True
        elif visited[i] == -1:
            return False
            
        visited[i] = -1 #방문 중
        
        for j in ref[i]:
            if not self.dfs(ref, visited, j):
                return False
        visited[i] = 1
        return True
        
            




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


#56. Merge Intervals
class Solution56:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
        Output: [[1,6],[8,10],[15,18]]
        
        Input: intervals = [[1,4],[4,5]]
        Output: [[1,5]]
        """
        intervals.sort(key= lambda x: x[0])
        
        i=1
        while(i<len(intervals)):
            if(intervals[i][0] <= intervals[i-1][1] and intervals[i][1] >= intervals[i-1][1]):                
                intervals[i-1] = [intervals[i-1][0], intervals[i][1]]
                intervals.pop(i)
            elif(intervals[i][0] <= intervals[i-1][1] and intervals[i][1] < intervals[i-1][1]):
                intervals.pop(i)
            else:
                i += 1
                
        return intervals


    """
    intervals [[1, 3], [2, 6], [8, 10], [15, 18]]
intervals.sort [[1, 3], [2, 6], [8, 10], [15, 18]]

interval = [1,3]
merged =[]
not merged:
	merged =[ [1,3] ]

interval =[2,6]
merged = [ [1,3] ]
merged[-1][-1] = 3 > interval[0] = 2:
	merged[-1][-1] = max(merged[-1][-1] = 3 ,interval[-1] = 6) =6
merged = [[1,6]]
    """

    def merge1(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key =lambda x: x[0])
        merged =[]
        for i in intervals:
			# if the list of merged intervals is empty 
			# or if the current interval does not overlap with the previous,
			# simply append it.
            if not merged or merged[-1][-1] < i[0]:
                merged.append(i)
			# otherwise, there is overlap,
			#so we merge the current and previous intervals.
            else:
                merged[-1][-1] = max(merged[-1][-1], i[-1])
        return merged
        """
        Time complexity:
            In python, use sort method to a list costs O(nlogn), where n is the length of the list.
            The for-loop used to merge intervals, costs O(n).
            O(nlogn)+O(n) = O(nlogn)
            So the total time complexity is O(nlogn).
        Space complexity:
            The algorithm used a merged list and a variable i.
            In the worst case, the merged list is equal to the length of the input intervals list. 
            So the space complexity is O(n), where n is the length of the input list.
        """
    def merge2(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        merged = []
        for i in range(len(intervals)):
            if merged == []:
                merged.append(intervals[i])
            else:
                previous_end = merged[-1][1]
                current_start = intervals[i][0]
                current_end = intervals[i][1]
                if previous_end >= current_start: # overlap
                    merged[-1][1] = max(previous_end,current_end)
                else:
                    merged.append(intervals[i])
        return merged


#227. Basic Calculator II
class Solution227:
    def calculate(self, s: str) -> int:
        """
        Input: s = " 3/2 "
        Output: 1
        
        Input: s = " 3+5 / 2 "
        Output: 5
        """
        num, stack, sign = 0, [], "+"
        for i in range(len(s)):
            if s[i].isdigit():
                num = num * 10 + int(s[i])
            if s[i] in "+-*/" or i == len(s) - 1:
                if sign == "+":
                    stack.append(num)
                elif sign == "-":
                    stack.append(-num)
                elif sign == "*":
                    stack.append(stack.pop()*num)
                else:
                    stack.append(int(stack.pop()/num))
                num = 0
                sign = s[i]
        return sum(stack)


#973. K Closest Points to Origin
class Solution973:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Input: points = [[1,3],[-2,2]], k = 1
        Output: [[-2,2]]
        
        Input: points = [[3,3],[5,-1],[-2,4]], k = 2
        Output: [[3,3],[-2,4]]
        
        Runtime: 616 ms, faster than 96.97%
        Memory Usage: 19.4 MB, less than 99.08%
        """
        points.sort(key = lambda x: x[0]**2+x[1]**2)
        return points[:k]




#476. Number Complement
class Solution476:
    def findComplement(self, num: int) -> int:
        """
        The integer 5 is "101" in binary and its complement is "010" which is the integer 2.
        """
        two = [2**i for i in range(30,-1,-1)]
        for i in range(31):
            if num >= 2**(30-i):
                two[i] = 0
                num -= 2**(30-i)
        
        start = two.index(0)
        return reduce(lambda x,y: x+y , two[start:], 0)



    def findComplement1(self, num):
        mask = 1 << (len(bin(num)) - 2)
        return (mask - 1) ^ num


    def findComplement2(self, num):
        i = 1
        while i <= num:
            i = i << 1
        return (i - 1) ^ num


#1015. Smallest Integer Divisible by K
class Solution1015:
    def smallestRepunitDivByK(self, k: int) -> int:
        """
        There is a x such that x%k ==0. 
        pf) Let N(n) = 11...1(nth of 1). Then N1... %k are in [1 ~ k-1]. 
            If Nj%k == Ni%k, then Nj - Ni % k == 0 and 
            it means N(j-i)%k==0. (-><-)
        """
        if(k%10 not in [1,3,7,9]): return -1
        num = 1
        while(1):
            if(num%k == 0): 
                return len(str(num))
            else:
                num = num*10 +1
                
    
    def smallestRepunitDivByK1(self, k: int) -> int:    
        if(k%10 not in [1,3,7,9]): return -1
        module = 1
        for Ndigits in range(1, k+1):
            if module % k == 0:
                return Ndigits
            module = (module * 10 + 1) % k
        



#1026. Maximum Difference Between Node and Ancestor
class Solution1026:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        """
        Input: root = [8,3,10,1,6,null,14,null,null,4,7,13]
        Output: 7
        
        Input: root = [1,null,2,null,0,3]
        Output: 3
        
        Runtime: 2585 ms, faster than 5.08%
        Memory Usage: 167.5 MB, less than 5.08%
        """
        
        def dfs(h,a):
            if not h:
                return 0
            out = dfs(h.left,a[:]+[h.val])
            out = max(out, dfs(h.right,a[:]+[h.val]))
            for anc in a:
                out = max(abs(h.val - anc), out)
            return out
        
        return dfs(root,[])
        
    
    def maxAncestorDiff1(self, root: Optional[TreeNode]) -> int:
        """
        Runtime: 217 ms, faster than 7.04%
        Memory Usage: 167.5 MB, less than 5.08%
        """
        #끝에만 가서 해보면 되겠는데?
        def gotoend(h,a):
            out = 0
            if(h.left):
                out = gotoend(h.left,a[:]+[h.val])
            if(h.right):
                out = max(out, gotoend(h.right,a[:]+[h.val]))
            
            if not h.left and not h.right:
                a.append(h.val)
                out = max(a) - min(a)
                return out
            else:
                return out
        
        return gotoend(root,[])


    def maxAncestorDiff2(self, root):
        """
        Runtime: 32 ms, faster than 94.74%
        Memory Usage: 19.3 MB, less than 87.08%
        """
        if not root: return 0
        return self.helper(root, root.val, root.val)
    
    def helper(self, node, high, low):
        if not node:
            return high - low
        high = max(high, node.val)
        low = min(low, node.val)
        return max(self.helper(node.left, high, low), self.helper(node.right,high,low))



    def maxAncestorDiff3(self, root):
        if not root: return 0
        
        def helper(node, high, low):
            if not node:
                return high - low
            high = max(high, node.val)
            low = min(low, node.val)
            return max(helper(node.left, high, low), helper(node.right,high,low))
    
        return helper(root, root.val, root.val)




#1010. Pairs of Songs With Total Durations Divisible by 60
class Solution1010:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        """
        Input: time = [30,20,150,100,40]
        Output: 3
        
        Input: time = [60,60,60]
        Output: 3
        
        Runtime: 345 ms, faster than 12.72%
        Memory Usage: 17.9 MB, less than 48.64%
        """
        dic, output = collections.defaultdict(int), 0
        for num in time:
            dic[num%60] += 1
        for i in range(1,30):
            output += dic[i]*dic[60-i]
        output += int((dic[0] * (dic[0]-1))/2)
        output += int((dic[30] * (dic[30]-1))/2)
        return output


#131. Palindrome Partitioning
class Solution131:
    """
    Input: s = "aab"
    Output: [["a","a","b"],["aa","b"]]
    
    
    aab, [], []
    a is pal -> dfs(ab, [a], []). a is pal -> dfs(b,[a,a],[]). bispal -> dfs(,[a,a,b],[]). res.aapend([a,a,b])
    aa is pal -> dfs(b, [aa], []). bis pal -> dfs(, [aa, b], [])
    
    Runtime: 1105 ms, faster than 7.02%
    Memory Usage: 26.4 MB, less than 98.88%
    """
    def partition(self, s: str) -> List[List[str]]:
        def ispal(s):
            return(s == s[::-1])
    
        def dfs(s, path, res):
            if not s:
                res.append(path)
                return
            for i in range(1, len(s)+1):
                if ispal(s[:i]):
                    dfs(s[i:], path+[s[:i]], res)
        
        res = []
        dfs(s,[],res)
        return res


#1094. Car Pooling
class Solution1094:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        """
        Input: trips = [[2,1,5],[3,3,7]], capacity = 4
        Output: false
        
        Input: trips = [[2,1,5],[3,3,7]], capacity = 5
        Output: true
        
        Runtime: 540 ms, faster than 5.19%
        Memory Usage: 14.7 MB, less than 77.09%
        """
        cap = [capacity]*max(trips, key=lambda x: x[2])[2]
        for trip in trips:
            for num in range(trip[1],trip[2]):
                cap[num] -= trip[0]
                if(cap[num]<0): return False
        
        return True



class Solution382:
    """
    Runtime: 94 ms, faster than 43.46%
    Memory Usage: 17.5 MB, less than 8.66%
    """
    
    def __init__(self, head: Optional[ListNode]):
        dummy,length = head,[],0
        while(dummy):
            length += 1
            dummy = dummy.next
        self.head = head
        self.length = length
    
    def getRandom(self) -> int:
        num,dum = random.randint(0, self.length - 1),self.head
        for i in range(num):
            dum = dum.next
        return dum.val


#1041. Robot Bounded In Circle
class Solution1041:
    def isRobotBounded(self, instructions: str) -> bool:
        """
        Runtime: 38 ms, faster than 25.22%
        Memory Usage: 14.4 MB, less than 15.52%
        """
        if "G" not in instructions:
            return True
        direction,position = 0,[0,0]
        
        def changeDirection(string,direction):
            if string == "L":
                direction = direction - 1 if direction > 0 else 3
            else :
                direction = direction + 1 if direction < 3 else 0
            
            return direction
        
        def changePosition(instruction, position, direction):
            if direction == 0: position[1] += 1
            elif direction == 1: position[0] += 1
            elif direction == 2: position[1] -= 1
            else: position[0] -= 1
            
            return position
        
        for instruction in instructions:
            if instruction == "G":
                position = changePosition(instruction,position,direction)
                if position == [0,0]: return True
            else:
                direction = changeDirection(instruction,direction)
        return False
            
        
#701. Insert into a Binary Search Tree
class Solution701:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if(root == None): return TreeNode(val);
        if(root.val < val): root.right = self.insertIntoBST(root.right, val);
        else: root.left = self.insertIntoBST(root.left, val);
        return(root)
        
#452. Minimum Number of Arrows to Burst Balloons
class Solution452:
    """
    Runtime: 1765 ms, faster than 27.60%
    Memory Usage: 59.2 MB, less than 35.36%
    
    Input: points = [[10,16],[2,8],[1,6],[7,12]]
    Output: 2
    """
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda x: x[1])
        count,end = 1,points[0][1]
        for point in points:
            if point[0] > end:
                count += 1
                end = point[1]
        return count


#8. String to Integer (atoi)
class Solution8:
    def myAtoi(self, s: str) -> int:
        """
        왜 비추가 추천의 3배인지 알 수 있는 문제 ㅡ
        Runtime: 45 ms, faster than 29.95%
        Memory Usage: 14.1 MB, less than 81.95%
        """
        digits=""
        s = s.strip()
        if(s=="" or s[0] not in "0123456789-+"): return 0
        elif s[0]=='-': 
            digits,s='-',s[1:]
        elif s[0]=='+' :
            s=s[1:]
        for string in s:
            if(string.isdigit()):
                digits += string
            else: break
        
        if (digits=='-' or digits=='') : return 0
        digits = int(digits)
        if digits < -2**31 :
            return -2**31                
        elif digits > 2**31 -1: 
            return 2**31 -1
        else:
            return digits



#849. Maximize Distance to Closest Person
class Solution849:
    def maxDistToClosest(self, seats: List[int]) -> int:
        """
        Runtime: 128 ms, faster than 89.20%
        Memory Usage: 14.5 MB, less than 92.48%
        """
        output,length = 0,0        
        for seat in seats:
            if seat==0 : 
                length += 1
            else:
                output = max(output,length)
                length = 0
        output = (output-1)//2 + 1
        length = 0
        #first and last
        if(seats[0] == 0):
            for seat in seats:
                if seat==0 : 
                    length += 1
                else:
                    output = max(output,length)
                    length = 0
                    break
        if(seats[-1] == 0):
            for seat in seats[::-1]:
                if seat==0 : 
                    length += 1
                else:
                    output = max(output,length)
                    length = 0
                    break
        return output



#142. Linked List Cycle II
class Solution142:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Runtime: 67 ms, faster than 35.07%
        Memory Usage: 17.3 MB, less than 56.58%
        """
        if(not head):
            return
        slow,fast = head,head
        while(fast.next and fast.next.next):
            slow = slow.next
            fast = fast.next.next
            if(slow == fast):
                if(slow == head):
                    return slow
                slow = head
                while(fast.next and fast.next.next):
                    slow = slow.next
                    fast = fast.next
                    if(slow == fast):
                        return slow
        return 


#875. Koko Eating Bananas
class Solution875:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        """
        Runtime: 871 ms, faster than 13.29%
        Memory Usage: 15.6 MB, less than 18.51%
        """
        import math
        l,r = 1,max(piles)
        while(l<r):
            m = (l+r)//2
            hour = 0
            for pile in piles:
                hour += math.ceil(pile/m)
            if(hour <= h):
                r=m
            else:
                l=m+1
        return l


#134. Gas Station
class Solution134:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        """
        Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
        Output: 3
        
        Runtime: 3498 ms, faster than 5.04%
        Memory Usage: 19 MB, less than 16.63%
        """
        #가스량이 적으면 불가능, 같으면 더해가면서확인, 더크면 유니크솔루션이므로 무조건 맥스값에서 시작해야함.
        #나에게 취한다
        if(sum(gas) < sum(cost)): 
            return -1
        elif(sum(gas) == sum(cost)):
            length = len(gas)
            for index in range(length):
                energy = 0
                for i in range(index,length+index):
                    if(i>=length):
                        i %= length
                    if(energy>= 0):
                        energy = energy + gas[i] - cost[i]
                    else:
                        break
                if energy < 0:
                    continue
                else:
                    break
            return index
        else:
            return gas.index(max(gas))

    def canCompleteCircuit1(self, gas: List[int], cost: List[int]) -> int:
        """
        [5,1,2,3,4]
        [4,4,1,5,1]
        Runtime: 666 ms, faster than 30.01%
        Memory Usage: 18.7 MB, less than 94.41%
        """
        sum_gas, sum_cost = sum(gas), sum(cost)
        if(sum_gas < sum_cost): 
            return -1
        elif(sum_gas == sum_cost):
            #순간 연료합이 음수인 다음부터 시작하는것으로 초기화.
            i,energy = 0,0
            for index in range(len(gas)):
                energy = energy + gas[index] - cost[index]                
                if energy < 0:
                    energy = 0
                    i = index + 1
                    continue
            return i
        else:
            return gas.index(max(gas))

    def canCompleteCircuit2(self, gas: List[int], cost: List[int]) -> int:
        #유니크 솔루션이기때문에...
        trip_tank, curr_tank, start, n = 0, 0, 0, len(gas)
        for i in range(n):
            trip_tank += gas[i] - cost[i]
            curr_tank += gas[i] - cost[i]
            if curr_tank < 0:
                start = i + 1
                curr_tank = 0 
        return start if trip_tank >= 0 else -1


#1291. Sequential Digits
class Solution1291:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        """
        Runtime: 55 ms, faster than 8.73%
        Memory Usage: 14.4 MB, less than 22.22%
        """
        digits, nums = '123456789',[]
        for window_length in range(1,10):
            for i in range(0,10-window_length):
                nums.append(int(digits[i:i+window_length]))                
        return [num for num in nums if num >= low and num <= high]

    def sequentialDigits1(self, low: int, high: int) -> List[int]:
        """
        Runtime: 23 ms, faster than 96.83%
        Memory Usage: 14.1 MB, less than 80.16%
        """
        nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 23, 34, 45, 56, 67, 78, 89, 123, 234, 345, 456, 567, 678, 789, 1234, 2345, 3456, 4567, 5678, 6789, 12345, 23456, 34567, 45678, 56789, 123456, 234567, 345678, 456789, 1234567, 2345678, 3456789, 12345678, 23456789, 123456789]
        return [num for num in nums if num >= low and num <= high]


#1305. All Elements in Two Binary Search Trees
class Solution1305:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        """
        Runtime: 418 ms, faster than 52.35%
        Memory Usage: 22.5 MB, less than 45.36%
        """
        output = []
        def dfs(node):
            if not node : 
                return
            else :
                output.append(node.val)
                if(node.left):
                    dfs(node.left)
                if(node.right):
                    dfs(node.right)
        dfs(root1)
        dfs(root2)
        return sorted(output)
        

#421. Maximum XOR of Two Numbers in an Array
class Solution421:
    def findMaximumXOR(self, nums: List[int]) -> int:        
        output = 0
        nums = list(set(nums))
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                output = max(nums[i]^nums[j],output)
        return output



    def findMaximumXOR1(self, nums: List[int]) -> int:
        """
        풀이봐도 모르겠다. 다음에 다시풀자. 맨왼쪽비트부터 마스킹해서 어쩌구저쩌구;;
        """
        answer = 0
        for i in range(32)[::-1]:
            answer <<= 1
            prefixes = {num >> i for num in nums}
            answer += any(answer^1 ^ p in prefixes for p in prefixes)
        return answer



#525. Contiguous Array
class Solution525:
    def findMaxLength(self, nums: List[int]) -> int:
        import collections
        g = collections.defaultdict(list)
        g[0].append(-1)
        output = 0
        for i,num in enumerate(nums):
            if(num) :
                output += 1
            else:
                output -= 1
            g[output].append(i)
        output = 0
        for value in g.values():
            output = max(output, max(value)-min(value))
        return output


#80. Remove Duplicates from Sorted Array II
class Solution80:
    def removeDuplicates(self, nums: List[int]) -> int:
        n,f,k = None,0,-1
        for num in nums:
            if(n==num):
                f += 1
            else:
                n = num
                f = 1
            if(f<3):
                k += 1
            nums[k] = n
        return k+1


#532. K-diff Pairs in an Array
class Solution532:
    def findPairs(self, nums: List[int], k: int) -> int:
        if(k==0):
            nums.sort()
            length = len(nums)        
            if(length < 2):
                return 0
            i,num,output = 1,nums[0],0
            while(i < length):
                if(nums[i] != num):
                    num = nums[i]
                    i += 1
                else:
                    output += 1
                    while(num == nums[i] and i < length - 1):
                        i+=1
                    num = nums[i]
                    i += 1                    
        else:
            nums = sorted(list(set(nums)))
            length = len(nums)        
            if(length < 2):
                return 0
            i,j,output = 0,1,0
            while(j < length):
                while(i<j):
                    if(nums[j] - nums[i] < k):
                        break
                    elif(nums[j] - nums[i] == k):
                        i+=1
                        output += 1
                    else:
                        i+=1
                j += 1
        return output


#디스커스 짱!!!
    def findPairs1(self, nums, k):
        res = 0
        c = collections.Counter(nums)
        for i in c:
            if k > 0 and i + k in c or k == 0 and c[i] > 1:
                res += 1
        return res

    def findPairs2(self, nums, k):
        c = collections.Counter(nums)
        return  sum(k > 0 and i + k in c or k == 0 and c[i] > 1 for i in c)


#560. Subarray Sum Equals K
class Solution560:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # window,length,output = 1,len(nums),0
        # while(window <= length):
        #     for i in range(length+1-window):
        #         if(sum(nums[i:i+window]) == k):
        #             output += 1
        #     window += 1
        # return output
        
        #출발점에서부터 현재지점 직전까지 합을 테이블에 저장해두자.
        # length,output = len(nums),0
        # table = [0]*1
        # for i in range(length):
        #     now = nums[i]
        #     output += table.count(k-now)
        #     table = [n+now for n in table] + [0]
        # return output
        
        #디스커스 runningsum 개념 참고
        from collections import defaultdict 
        summ,output = 0,0
        dic = defaultdict(int)
        dic[0] = 1
        for i in range(len(nums)):
            summ += nums[i]
            output += dic[summ-k]
            dic[summ] += 1
        return output


#78. Subsets
class Solution78:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        output = []
        for i in range(len(nums)+1):
            output += list(itertools.combinations(nums,i))
        return output


#24. Swap Nodes in Pairs
class Solution24:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if (not head or not head.next):
            return head
        node,next_node = head,head.next
        output = next_node
        while(next_node):
            tmp = next_node.next
            if(tmp):
                if(tmp.next):
                    node.next,next_node.next = next_node.next.next, node
                else:
                    node.next,next_node.next = tmp, node
                node,next_node = tmp,tmp.next
            else:
                node.next,next_node.next = tmp, node
                node,next_node = None,None
        return output


#39. Combination Sum
class Solution39:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        queue, output = collections.deque([[num] for num in candidates]), []
        while(queue):
            que = queue.popleft()
            if(sum(que) == target):
                output.append(que)
            for n in candidates[candidates.index(que[-1]):]:
                if(sum(que) + n <= target):
                    queue.append(que+[n])
        return output

    def combinationSumDFS(self, candidates: List[int], target: int) -> List[List[int]]:
        output = []
        def dfs(l,t):
            if t == 0:
                return output.append(l)
            elif t > 0:
                for n in candidates[candidates.index(l[-1]):]:
                    dfs(l+[n], t-n)
        for num in candidates:
            dfs([num],target-num)
        return output


#402. Remove K Digits
class Solution402:
    def removeKdigits(self, num: str, k: int) -> str:
        if len(num)==k :
            return "0"
        stack = []
        for digit in num :
            while k>0 and stack and stack[-1] > digit:
                k -= 1
                stack.pop()
            stack.append(digit)
        if k>0:
            stack = stack[:-k]
        return str(int(''.join(stack)))


#1288. Remove Covered Intervals
class Solution1288:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x : (x[0] - x[1]))
        i , length = 0,len(intervals)
        while(i < length):
            l,r = intervals[i][0],intervals[i][1]
            intervals = [[l,r]] + [interval for interval in intervals if interval[0]<l or interval[1]>r]
            i+=1
            length = len(intervals)
        return len(intervals)






























