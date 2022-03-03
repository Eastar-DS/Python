N = int(input())
nums = list(map(int,input().split()))
dp = nums[:]
for i in range(N):
    for j in range(i):
        if nums[j] < nums[i] and dp[j]+nums[i] > dp[i]:
            dp[i] = dp[j]+nums[i]
print(max(dp))