N = int(input())
nums = list(map(int,input().split()))
dp = [1]*N
for i in range(N):
    for j in range(i):
        if nums[j] < nums[i] and dp[j]+1 > dp[i]:
            dp[i] = dp[j]+1
print(max(dp))