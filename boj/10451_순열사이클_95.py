def dfs(i):
    num = nums[i]
    nums[i] = 0
    if nums[num]:
        dfs(num)

import sys
# input = sys.stdin.readline
for _ in range(int(input())):
    N = int(input())
    nums = [0] + list(map(int,input().split()))
    output = 0
    for i in range(1,N+1):
        if nums[i]:
            dfs(i)
            output += 1
    print(output)