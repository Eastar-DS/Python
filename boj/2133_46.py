N = int(input())
if N%2 == 1:
    print(0)
else:
    N //= 2
    dp = [3] + [2]*(N-1)
    for i in range(1,N):
        dp[i] += 3*dp[i-1]
        for j in range(2,i+1):
            dp[i] += 2*dp[i-j]
    print(dp[-1])