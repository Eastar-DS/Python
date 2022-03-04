n,k = map(int,input().split())
dp = [[1]*(k) for _ in range(n+1)]
for i in range(1,n+1):
    for j in range(1,k):
        dp[i][j] = dp[i][j-1] + dp[i-1][j]
print(dp[-1][-1]%1000000000)


#   k
# n 1 2 3 4 5 6 7 8 9
# 1 1 2 3 4 5 6 7 8 9
# 2 1 3 6
# 3 1 4 10
# 4 1 5 
# 5 1 6 
# 6 1 7 
# 7 1 8 
# 8 1 9 
# 9 1 