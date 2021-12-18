import collections
import heapq
import functools
import itertools
import re
import sys
import math
import bisect
from typing import *

class Solution902:    
    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
        """
        Runtime: 71 ms, faster than 6.48% 
        Memory Usage: 14.6 MB, less than 8.33%
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
            
            