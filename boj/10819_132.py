#1
from itertools import permutations
N = int(input())
nums = list(map(int,input().split()))

output = 0
for per in permutations(nums,N):
    tmp = 0
    for i in range(N-1):
        tmp += abs(per[i] - per[i+1])
    output = max(tmp,output)
print(output)