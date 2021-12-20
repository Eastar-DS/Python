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
        





















