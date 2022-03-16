#1 stack
stack,stick,output = [],0,0
ps = [0] + list(input())
for i in range(1,len(ps)):
    if ps[i] == '(':
        stack.append(ps[i])
        stick += 1
    else:
        stack.pop()
        stick -= 1
        if ps[i-1] == '(':            
            output += stick
        else:
            output += 1
print(output)

#2 stack 숫자로
stick,output = 0,0
ps = list(input())
for i in range(len(ps)):
    if ps[i] == '(':
        stick += 1
    else:
        stick -= 1
        if ps[i-1] == '(':
            output += stick
        else:
            output += 1
print(output)

#3 레이저 0으로
s,o,p=0,0,list(input().replace('()','0'))
for i in range(len(p)):
    if p[i]=='(':
        s+=1
    elif p[i]=='0':
        o+=s
    else:
        s-=1
        o+=1
print(o)