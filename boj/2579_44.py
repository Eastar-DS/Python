N = int(input())
nums = []
for _ in range(N):
    nums.append(int(input()))
if N<3:
    print(sum(nums))
elif N==3:
    print(nums[2]+max(nums[0], nums[1]))
else:
    dp = [nums[0],nums[0]+nums[1],nums[2]+max(nums[0], nums[1])]
    for i in range(3,N):
        dp.append(max(dp[i-3]+nums[i-1]+nums[i], dp[i-2]+nums[i]))
    print(dp[-1])

#dp[i] = dp[i-3]+nums[i-1]+nums[i] or dp[i-2]+nums[i]