class Solution(object):
    def isValid(s):
        """
        :type s: str
        :rtype: bool
        ']' error -> stack에 0을 채워넣는 방법으로 해결
        Runtime: 16 ms, faster than 88.40% of Python online submissions for Valid Parentheses.
        Memory Usage: 13.5 MB, less than 88.30% of Python online submissions for Valid Parentheses.
        """
        stack = [0]
        idx = 0
        dic = {')':'(', '}':'{', ']':'['}
        if(len(s) == 0):
            return True
        for string in s:
            if(string == '(' or string == '{' or string == '['):
                stack.append(string)
                idx += 1
            else:
                if(dic[string] == stack[idx] ):
                    del stack[idx]
                    idx -= 1
                else:
                    return False
        if(idx == 0):
            return True
        else:
            return False
    
    
    def evalRPN(tokens):
        """
        :type tokens: List[str]
        :rtype: int
        ["10","6","9","3","+","-11","*","/","*","17","+","5","+"] : 파이썬3에서는 돌아가는데 Leetcode 파이썬에서는 안돌아감
        Runtime: 60 ms, faster than 93.10% of Python3 online submissions for Evaluate Reverse Polish Notation.
        Memory Usage: 14.8 MB, less than 9.89% of Python3 online submissions for Evaluate Reverse Polish Notation.
        """
        stack = [0]
        idx = 0
        for element in tokens:
            if(element == '+'):
                val = stack[idx - 1] + stack[idx]
                del stack[idx]
                del stack[idx - 1]
                idx -= 1
                stack.append(val)
            elif(element == '-'):
                val = stack[idx - 1] - stack[idx]
                del stack[idx]
                del stack[idx - 1]
                idx -= 1
                stack.append(val)
            elif(element == '*'):
                val = stack[idx - 1] * stack[idx]
                del stack[idx]
                del stack[idx - 1]
                idx -= 1
                stack.append(val)
            elif(element == '/'):
                val = stack[idx - 1] / stack[idx]
                del stack[idx]
                del stack[idx - 1]
                idx -= 1
                stack.append(int(val))
            else:
                stack.append(int(element))
                idx += 1
        return stack[idx]
        
#stack은 뭔가 싫어요숫자가 많네...            
            
            #asdf
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            