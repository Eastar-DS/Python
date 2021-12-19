import collections
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from typing import *

#902. Numbers At Most N Given Digit Set
class Solution902:    
    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
        """
        Runtime: 40 ms, faster than 25.00%
        Memory Usage: 14.4 MB, less than 50.93%
        이것이... 하드의 벽?
        """
        #the number of (len(n)-1)length
        r,output = len(digits),0
        for i in range(1,len(str(n))):
            output += r**i
        
        
        def find(num,first):
            temp = [digit for digit in digits if int(digit) <= int(first)]
            out = 0
            if(first in temp):
                if(len(num)>1):
                    out += find(num[1:], num[1])
                    out += (len(temp) - 1) * ((len(digits))**(len(num)-1) )
                else:
                    out += len(temp)
            else:
                out = len(temp)*((len(digits))**(len(num)-1) )
            return out
        
        output += find(str(n), str(n)[0])
        return output
            
            
    
    
    def atMostNGivenDigitSet1(self, D, N):
        """
            N has n digits, so all numbers less than n digits are valid, 
                            which are: sum(len(D) ** i for i in range(1, n))
            The loop is to deal with all numbers with n digits, 
                                    considering from N[0], N[1] back to N[n-1]. 
            For example, N[0] is valid only for c in D if c <= N[0]. If c < N[0], 
                then N[1], ..., N[n-1] can take any number in D but if c == N[0], 
                        then we need consider N[1], and the iteration repeats. 
            That's why if N[i] not in D, then we don't need to repeat the loop anymore.
            Finally i==n is addressed at the end when there exists all c in D that matches N
            나랑 똑같은 원리로 풀었는데 이렇게 깔끔하고 멋있게 풀다니...
            
            Runtime: 42 ms, faster than 22.22%
            Memory Usage: 14.4 MB, less than 50.93%
        """
        N = str(N)
        n = len(N)
        res = sum(len(D) ** i for i in range(1, n))
        i = 0
        while i < len(N):
            #sum(c < N[i] for c in D) is sum of bool(c < N[i])
            res += sum(c < N[i] for c in D) * (len(D) ** (n - i - 1))
            if N[i] not in D: break
            i += 1
        return res + (i == n)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    