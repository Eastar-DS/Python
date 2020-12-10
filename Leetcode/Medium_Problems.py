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
           
         




    def threeSum(nums):
            """
            :type nums: List[int]
            :rtype: List[List[int]]
            """
            out = []
            # dic = []
            size = len(nums)
            if(size < 3):
                return out
            
            for i in range(size-2):
                for j in range(size-2-i):
                    for k in range(size-2-(i+j)):
                        val = nums[i]+nums[i+j+1]+nums[i+j+k+2]
                        if(val == 0):
                            out.append([nums[i],nums[i+j+1],nums[i+j+k+2]])            
            
            return out
            



































        