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
    def addTwoNumbers1(l1, l2):
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
        Runtime: 16 ms, faster than 85.61% of Python online submissions for Swap Nodes in Pairs.
        Memory Usage: 13.2 MB, less than 99.94% of Python online submissions for Swap Nodes in Pairs.
        Follow up: Can you solve the problem without modifying the values in the list's nodes?
                    (i.e., Only nodes themselves may be changed.)
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
            while(node1.next != None):                
                val1 = node1.next.val
                val2 = node1.next.next.val
                print(val1, '  ', val2)
                node1.next.val = val2
                node1 = node1.next
                node1.next.val = val1
                node1 = node1.next
        else:
            idx = 0
            while(idx < length - 1):
                val1 = node1.next.val
                val2 = node1.next.next.val
                node1.next.val = val2
                node1 = node1.next
                node1.next.val = val1
                node1 = node1.next
                idx += 2
                
        return node2.next
            
    
    
    def rotateRight(head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        Runtime: 36 ms, faster than 13.58% of Python online submissions for Rotate List.
        Memory Usage: 13.5 MB, less than 62.57% of Python online submissions for Rotate List.
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
        for num in range(k % length):           
            for i in range(length):
                node2 = node2.next
            node2.next = node1.next
            node1.next = node2
            for j in range(length - 1):
                node2 = node2.next
            node2.next = None
            node2 = node1
            
        return node1.next

    
    def deleteDuplicates(head):
        """
        :type head: ListNode
        :rtype: ListNode
        []
        [1,1,2,3,3]
        [1]
        Runtime: 24 ms, faster than 94.19% of Python online submissions for Remove Duplicates from Sorted List.
        Memory Usage: 13.4 MB, less than 91.02% of Python online submissions for Remove Duplicates from Sorted List.
        """
        if(head == None):
            return None
        if(head.next == None):
            return head
        node1 = head
        node2 = ListNode(0)
        res = node2
        
        while((node1 != None) and (node1.next != None)):
            while((node1.next != None) and (node1.val == node1.next.val)):
                node1 = node1.next
            node2.next = node1
            node1 = node1.next            
            node2 = node2.next
        
        return res.next

    
    
    def deleteDuplicates2(head):
        """
        :type head: ListNode
        :rtype: ListNode
        [1,1]
        [1,1,1,2,3] -> [2]
        Runtime: 40 ms, faster than 8.49% of Python online submissions for Remove Duplicates from Sorted List II.
        Memory Usage: 13.5 MB, less than 31.27% of Python online submissions for Remove Duplicates from Sorted List II.
        """
        if(head == None):
            return None
        if(head.next == None):
            return head
        node1 = head
        node2 = ListNode(0)
        node3 = node2
        val = head.val
        while(node1.next != None):
            if(val == node1.next.val):
                while((node1.next != None) and (node1.val == node1.next.val)):
                    node1 = node1.next
                if(node1.next != None):
                    node1 = node1.next
                    val = node1.val
                    if(node1.next == None):
                        node2.next = ListNode(val)
                
            else:
                node2.next = ListNode(val)
                node2 = node2.next
                node1 = node1.next
                val = node1.val
                if(node1.next == None):
                        node2.next = ListNode(val)
        
        
        return node3.next
        
    
    
    def reverseList(head):
        """
        :type head: ListNode
        :rtype: ListNode
        한큐에 통과
        Runtime: 20 ms, faster than 92.18% of Python online submissions for Reverse Linked List.
        Memory Usage: 17.1 MB, less than 21.77% of Python online submissions for Reverse Linked List.
        """
        li = []
        node1 = head
        node2 = ListNode(0)
        res = node2
        while(node1 != None):
            li.append(node1.val)
            node1 = node1.next
            
        for i in range(len(li) - 1 , -1, -1):
            node2.next = ListNode(li[i])
            node2 = node2.next
        
        return res.next
            
    
    def reverseBetween(head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        한큐에 통과
        Runtime: 20 ms, faster than 65.66% of Python online submissions for Reverse Linked List II.
        Memory Usage: 13.5 MB, less than 87.26% of Python online submissions for Reverse Linked List II.
        """
        li1 = []
        li2 = []
        idx = 0
        node1 = head
        node2 = ListNode(0)
        res = node2
        
        while(node1 != None and idx < n):
            li1.append(node1.val)
            node1 = node1.next
            idx += 1
        
        for k in range(m-1):
            li2.append(li1[k])
        for i in range(n-1,m-2,-1):
            li2.append(li1[i])
        
        for j in range(n):
            node2.next = ListNode(li2[j])
            node2 = node2.next
        if(node1 != None):
            node2.next = node1
        
        return res.next
    
    
    def hasCycle(head):
        """
        :type head: ListNode
        :rtype: bool
        Follow up: Can you solve it using O(1) (i.e. constant) memory?
        어떻게 Cycle이 있는지 알 수 있지?
        slow와 fast의 도입생각을 못했음.
        Runtime: 36 ms, faster than 85.42% of Python online submissions for Linked List Cycle.
        Memory Usage: 19.7 MB, less than 63.99% of Python online submissions for Linked List Cycle.
        """
        if(head == None or head.next == None):
            return False
        slow = head
        fast = head.next
        while(slow != fast and fast != None and fast.next != None):
            slow = slow.next
            fast = fast.next.next
        if(slow == fast):
            return True
        else:
            return False
        
        
    
    def detectCycle(head):
        """
        :type head: ListNode
        :rtype: ListNode
        https://www.youtube.com/watch?v=LUm2ABqAs1w start지점을 어떻게 찾는건지 굉장히 놀라움.
        Floyd's cycle Detection algorithm
        Runtime: 40 ms, faster than 74.66% of Python online submissions for Linked List Cycle II.
        Memory Usage: 19.3 MB, less than 99.69% of Python online submissions for Linked List Cycle II.
        """
        if(head == None or head.next == None):
            return None
        slow = head.next
        fast = head.next.next
        while(fast != None and fast.next != None and slow != fast):
            slow = slow.next
            fast = fast.next.next
        
        if(slow == fast):
            slow = head
            while(slow != fast):
                slow = slow.next
                fast = fast.next
            return slow      
        
        else:
            return None
        
        
    def reorderList(head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        원큐
        Runtime: 88 ms, faster than 83.96% of Python online submissions for Reorder List.
        Memory Usage: 31.3 MB, less than 65.01% of Python online submissions for Reorder List.
        """
        node1 = head
        node2 = ListNode(0)
        length = 0
        list1 = []
        list2 = []
        while(node1 != None):
            list1.append(node1)
            node1 = node1.next
            length += 1
        
        for i in range(length - 1, (length- 1)//2, -1):
            list2.append(list1[i])
        
        if(length % 2 == 0):
            for i in range(length/2):
                node2.next = list1[i]
                node2 = node2.next
                node2.next = list2[i]
                node2 = node2.next
            node2.next = None    
        else:
            for i in range((length-1)/2):
                node2.next = list1[i]
                node2 = node2.next
                node2.next = list2[i]
                node2 = node2.next
            node2.next = list1[length//2]
            node2 = node2.next
            node2.next = None
                                
        
    def removeElements(head, val):
        """
        203. Linked List
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        1. [1], val = 1 인경우 오류(node1 = head로 뒀었음)
        2. return값을 head로 둬서 오류
        3. return값을 node1.next로 둬서 오류
        4. [1,1], val = 1(노드를 하나 줄여놓고 다음으로 넘어가버리니까 연산을 안해버림) : next로 넘어가는건 else경우로수
        Runtime: 68 ms, faster than 35.89% of Python online submissions for Remove Linked List Elements.
        Memory Usage: 20 MB, less than 96.61% of Python online submissions for Remove Linked List Elements.
        """
        node1 = ListNode(0)
        node1.next = head
        node2 = node1
        
        while(node1 != None and node1.next != None):
            val1 = node1.next.val
            if(val1 == val):
                node1.next = node1.next.next
            else:
                node1 = node1.next
        
        return node2.next
    
    
    def sortList(head):
        """
        148. Linked List, Sort
        :type head: ListNode
        :rtype: ListNode
        sort를 내가 구현한것이 아니라서 푼게 아니다.
        Runtime: 296 ms, faster than 85.98% of Python online submissions for Sort List.
        Memory Usage: 64.9 MB, less than 5.12% of Python online submissions for Sort List.
        """
        node1 = head
        node2 = ListNode(0)
        node3 = node2
        lis = []
        while(node1 != None):
            lis.append(node1.val)
            node1 = node1.next
        lis = sorted(lis)
        for i in range(len(lis)):
            node2.next = ListNode(lis[i])
            node2 = node2.next
            
        return node3.next
    
    
    def partition(head, x):
        """
        86. Linked List, Two Pointers
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        node5를 만들지 않고 했었음.
        Runtime: 20 ms, faster than 87.44% of Python online submissions for Partition List.
        Memory Usage: 13.4 MB, less than 90.23% of Python online submissions for Partition List.
        """
        node1 = head
        node2 = ListNode(0)
        node3 = ListNode(0)
        node4 = node2
        node5 = node3
        while(node1 != None):
            if(node1.val < x):
                node2.next = node1
                node1 = node1.next
                node2 = node2.next
            else:
                node3.next = node1
                node1 = node1.next
                node3 = node3.next
        node2.next = node5.next
        node3.next = None
        
        return node4.next
            
        
    def oddEvenList(head):
        """
        328. Linked List
        :type head: ListNode
        :rtype: ListNode
        Runtime: 28 ms, faster than 88.49% of Python online submissions for Odd Even Linked List.
        Memory Usage: 16.9 MB, less than 85.77% of Python online submissions for Odd Even Linked List.
        """
        node1 = head
        node2 = ListNode(0)
        node3 = ListNode(0)
        node4 = node2
        node5 = node3
        idx = 1
        while(node1 != None):
            if(idx % 2 == 1):
                node2.next = node1
                node1 = node1.next
                node2 = node2.next
                idx += 1
            else:
                node3.next = node1
                node1 = node1.next
                node3 = node3.next
                idx += 1
        node2.next = node5.next
        node3.next = None
        
        return node4.next
    
    
    def addTwoNumbers2(l1, l2):
        """
        445. Linked List
        Follow up: What if you cannot modify the input lists? In other words, reversing the lists is not allowed.
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        Runtime: 64 ms, faster than 69.27% of Python online submissions for Add Two Numbers II.
        Memory Usage: 13.7 MB, less than 19.35% of Python online submissions for Add Two Numbers II.
        """
        #Follow up으로 풀어보자.
        #각각 length 재주기
        len1 = l1
        len2 = l2
        length1 = 0
        length2 = 0
        while(len1 != None):
            len1 = len1.next
            length1 += 1
        while(len2 != None):
            len2 = len2.next
            length2 += 1
        #7243 + 564라면 7243 + 0564로 맞춰준다. node의 자릿수를 큰 length + 1로 만들어준다.
        nod = ListNode(0)
        node = nod
        node1 = ListNode(1)
        node2 = ListNode(1)
        node3 = node1
        node4 = node2        
        if(length1 > length2):
            idx = length1 - length2
            while(idx != 0):
                node2.next = ListNode(0)
                node2 = node2.next
                idx -= 1
            for i in range(length1):
                nod.next = ListNode(0)
                nod = nod.next
        else:
            idx = length2 - length1
            while(idx != 0):
                node1.next = ListNode(0)
                node1 = node1.next
                idx -= 1
            for i in range(length2):
                nod.next = ListNode(0)
                nod = nod.next
        node1.next = l1
        node2.next = l2
        #재귀함수를 통해 해결해보자!
        def f(nod1,nod2,node):
            carry = 0
            if(nod1.next != None):
                carry = f(nod1.next, nod2.next, node.next)
            #맨 끝자리부터 일어날 일들
            val1 = nod1.val
            val2 = nod2.val
            if(val1 + val2 + carry > 9):
                val3 = val1 + val2 + carry - 10
                carry = 1
            else:
                val3 = val1 + val2 + carry
                carry = 0
            
            node.val = val3
            
            return carry
        #시작
        carry = f(node3.next, node4.next, node.next)
        #carry가 1이면 최고자리숫자를 더해서 올려진것이므로 node.val을 바꿔준다.
        if(carry == 1):
            node.val = 1
            return node
        #carry가 0이면 node의 맨앞 0은 쓸모가 없어지므로
        else:
            return node.next
        
            
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    