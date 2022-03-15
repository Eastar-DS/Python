#1 dic
import sys
# input = sys.stdin.readline
size = int(input())
m,n = map(int,input().split())
A = [int(input()) for _ in range(m)]
B = [int(input()) for _ in range(n)]

aa,bb = {sum(A):1},{sum(B):1}
for i in range(-m,0):
    summ = A[i]
    aa[summ] = aa.get(summ,0)+1
    for j in range(i+1,i+m-1):
        summ+=A[j]
        aa[summ] = aa.get(summ,0)+1
for i in range(-n,0):
    summ = B[i]
    bb[summ] = bb.get(summ,0)+1
    for j in range(i+1,i+n-1):
        summ+=B[j]
        bb[summ] = bb.get(summ,0)+1

output = aa.get(size,0) + bb.get(size,0)
for s in aa:
    if size-s in bb:
        output += aa[s]*bb[size-s]
print(output)




#2 dic 1개만 좀더빨라짐
import sys
# input = sys.stdin.readline
size = int(input())
m,n = map(int,input().split())
A = [int(input()) for _ in range(m)]
B = [int(input()) for _ in range(n)]

aa= {sum(A):1}
for i in range(-m,0):
    summ = A[i]
    aa[summ] = aa.get(summ,0)+1
    for j in range(i+1,i+m-1):
        summ+=A[j]
        aa[summ] = aa.get(summ,0)+1

output = aa.get(size,0)
for i in range(-n,0):
    summ = B[i]
    if summ == size:
        output += 1
    elif size-summ in aa:
        output += aa[size-summ]
    for j in range(i+1,i+n-1):
        summ+=B[j]
        if summ == size:
            output += 1
        elif size-summ in aa:
            output += aa[size-summ]
summ = sum(B)
if summ == size:
    output += 1
elif size-summ in aa:
    output += aa[size-summ]
print(output)



#3 list만 써보자! 약간느리네;;?

import sys
# input = sys.stdin.readline
size = int(input())
m,n = map(int,input().split())
A = [int(input()) for _ in range(m)]
B = [int(input()) for _ in range(n)]

output = 0
aa,bb = [sum(A)],[sum(B)]
if aa[0] == size:
    output+=1
if bb[0] == size:
    output+=1
for i in range(-m,0):
    summ = A[i]
    aa.append(summ)
    if summ==size : output+=1
    for j in range(i+1,i+m-1):
        summ+=A[j]
        aa.append(summ)
        if summ==size : output+=1
for i in range(-n,0):
    summ = B[i]
    bb.append(summ)
    if summ==size : output+=1
    for j in range(i+1,i+n-1):
        summ+=B[j]
        bb.append(summ)
        if summ==size : output+=1
aa.sort()
bb.sort(reverse=True)
aa.append(2*(10**6)+1)
bb.append(2*(10**6)+1)

i,j = 0,0
laa,lbb = len(aa),len(bb)

while(i<laa and j<lbb):
    summ = aa[i]+bb[j]
    if summ>size:
        j+=1
    elif summ<size:
        i+=1
    else:
        x,y = i,j
        aaa,bbb = aa[i],bb[j]
        while(aa[i]==aaa):
            i+=1
        while(bb[j]==bbb):
            j+=1
        output += (i-x)*(j-y)
print(output)

















