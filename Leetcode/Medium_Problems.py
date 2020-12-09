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
        while(node1.next != None or node2.next != None):
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
        
        if(node1.next == None and node2.next == None):
            val = node1.val + node2.val + carry
            if(val > 9):
                carry = 1
                val -= 10
            else:
                carry = 0
            node3.next = ListNode(val)
            if(carry == 1):
                node3.next = ListNode(1)
            return l3
        
        else:
            if(node2 == None):
                node12 = node1
            else:
                node12 = node2
            
            
        
        
        
            
        # l3=[]
        # len1 = len(l1)
        # len2 = len(l2)
        # minlen = min(len1,len2)
        # carry1 = 0
        # for i in range(minlen):
        #     if(l1[i] + l2[i] + carry1 > 9):
        #         l3.append(l1[i]+l2[i] - 10 + carry1)
        #         carry1 = 1
        #     else:
        #         l3.append(l1[i]+l2[i] + carry1)
        #         carry1 = 0
        # if(len1 == len2):
        #     if(carry1 == 0):
        #         return l3
        #     if(carry1 == 1):
        #         l3.append(1)
        #         return l3
        # maxlen = max(len1,len2)
        # for j in range(maxlen - minlen):
        #     if(len1 > len2):
        #         if((l1[minlen + j] + carry1) < 10):
        #             l3.append(l1[minlen + j] + carry1)
        #             carry1 = 0
        #         else:
        #             l3.append(0)
        #             carry1 = 1
        #     else:
        #         if((l2[minlen + j] + carry1) < 10):
        #             l3.append(l2[minlen + j] + carry1)
        #             carry1 = 0
        #         else:
        #             l3.append(0)
        #             carry1 = 1
        # if(carry1 == 1):
        #     l3.append(1)
        # return l3
        