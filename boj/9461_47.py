#1
import sys
input = sys.stdin.readline
for _ in range(int(input())):
    N = int(input())
    if N <= 10:
        print([0,1,1,1,2,2,3,4,5,7,9][N])
    else:
        P = [0,1,1,1,2,2,3,4,5,7,9]
        for i in range(11,N+1):
            P.append(P[i-1]+P[i-5])
        print(P[-1])

#2
import sys
input = sys.stdin.readline
for _ in range(int(input())):
    N = int(input())
    if N <= 10:
        print([0,1,1,1,2,2,3,4,5,7,9][N])
    else:
        p1,a,b,c,p2 = 3,4,5,7,9
        for i in range(11,N+1):
            p1,a,b,c,p2 = a,b,c,p2,p1+p2
        print(p2)

#3
import sys
input = sys.stdin.readline
for _ in range(int(input())):
    N = int(input())
    if N <= 10:
        print([0,1,1,1,2,2,3,4,5,7,9][N])
    else:
        p1,*a,p2 = 3,4,5,7,9
        for i in range(11,N+1):
            p1,*a,p2 = *a,p2,p1+p2
        print(p2)