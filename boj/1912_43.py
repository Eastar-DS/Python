n = int(input())
nums = list(map(int,input().split()))
dp = nums[:]
for i in range(1,n):
    dp[i] = max(nums[i], dp[i-1]+nums[i])
print(max(dp))