#pypy
import math
N = int(input())
if N<10:
    print([0,1,2,3,1,2,3,4,2,1][N])
else:
    dp = [0]*(N+1)
    for i in range(1,N+1):
        tmp = 100000
        for j in range(1,math.floor(i**0.5)+1):
            if dp[i-j**2]+1 < tmp:
                dp[i] = dp[i-j**2]+1
                tmp = dp[i]
    print(dp[-1])

#2
N = int(input())
if N<10:
    print([0,1,2,3,1,2,3,4,2,1][N])
else:
    dp = [0]*(N+1)
    square = [i*i for i in range(1,317) if i*i<=N]
    for i in range(1,N+1):
        dp[i] = dp[i-1]+1
        for j in square:
            if i>=j and dp[i-j]+1 < dp[i]:
                dp[i] = dp[i-j]+1
    print(dp[-1])
    