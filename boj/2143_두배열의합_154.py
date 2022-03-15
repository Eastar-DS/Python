#1
import sys
# input = sys.stdin.readline

T = int(input())
n = int(input())
A = list(map(int,input().split()))
m = int(input())
B = list(map(int,input().split()))

aa,bb = {},{}
for i in range(n):
    summ = A[i]
    if summ in aa:
        aa[summ]+=1
    else:
        aa[summ]=1
    for j in range(i+1,n):
        summ += A[j]
        if summ in aa:
            aa[summ]+=1
        else:
            aa[summ]=1
for i in range(m):
    summ = B[i]
    if summ in bb:
        bb[summ]+=1
    else:
        bb[summ]=1
    for j in range(i+1,m):
        summ += B[j]
        if summ in bb:
            bb[summ]+=1
        else:
            bb[summ]=1
output = 0
for s in aa:
    if T-s in bb:
        output += aa[s]*bb[T-s]
print(output)




#2 dic1개쓰기
import sys
# input = sys.stdin.readline

T = int(input())
n = int(input())
A = list(map(int,input().split()))
m = int(input())
B = list(map(int,input().split()))

aa= {}
for i in range(n):
    summ = A[i]
    if summ in aa:
        aa[summ]+=1
    else:
        aa[summ]=1
    for j in range(i+1,n):
        summ += A[j]
        if summ in aa:
            aa[summ]+=1
        else:
            aa[summ]=1

output = 0
for i in range(m):
    summ = B[i]
    if T-summ in aa:
        output += aa[T-summ]
    for j in range(i+1,m):
        summ += B[j]
        if T-summ in aa:
            output += aa[T-summ]
print(output)

#3 부분합 다르게구하기 더느림.
import sys
# input = sys.stdin.readline

T = int(input())
n = int(input())
A = [0] + list(map(int,input().split()))
m = int(input())
B = [0] + list(map(int,input().split()))

for i in range(1,n+1):
    A[i] += A[i-1]
for i in range(1,m+1):
    B[i] += B[i-1]

aa= {}
for i in range(n):
    for j in range(i+1,n+1):
        summ = A[j]-A[i]
        if summ in aa:
            aa[summ]+=1
        else:
            aa[summ]=1

output = 0
for i in range(m):
    for j in range(i+1,m+1):
        summ = B[j]-B[i]
        if T-summ in aa:
            output += aa[T-summ]
print(output)