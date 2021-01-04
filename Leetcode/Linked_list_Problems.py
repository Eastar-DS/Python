# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:41:02 2020

@author: user
"""

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
    
    
    
    def removeNthFromEnd(head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        Runtime: 8 ms, faster than 99.92%
        Memory Usage: 13.4 MB, less than 73.46%
        """
        node1 = head        
        length = 1;
        node2 = ListNode(0)
        node2.next = head
        node3 = node2                
        
        idx = 0        
        while(node1.next != None):
            node1 = node1.next
            length += 1       
        delnum = length - n

        while(idx != (delnum)):
            idx += 1
            node2 = node2.next
        node2.next = node2.next.next
        
        return node3.next
        
        
    def swapPairs(head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if(head == None):
            return None
        lennode = head
        length = 1
        while(lennode.next != None):
            length += 1
            lennode = lennode.next
        node1 = ListNode(0)
        node1.next = head
        node2 = node1
        print(length)
        if(length % 2 == 0):
            while(node1 != None):                
                val1 = node1.next.val
                val2 = node1.next.next.val
                print(val1, '  ', val2)
                node1.next.val = val2
                node1 = node1.next
                node1.next = val1
                node1 = node1.next
        else:
            idx = 0
            while(idx < length - 1):
                val1 = node1.next.val
                val2 = node1.next.next.val
                node1.next.val = val2
                node1 = node1.next
                node1.next = val1
                node1 = node1.next
                idx += 2
                
        return node2.next
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    