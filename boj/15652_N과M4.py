#1
def dfs(index,nums):
    if index==M:
        return print(" ".join(map(str,nums)))
    tmp = 1
    if len(nums):
        tmp = nums[-1]
    for i in range(tmp,N+1):
        dfs(index+1,nums+[i])

N,M = map(int,input().split())
dfs(0,[])

#2 combinations
import itertools 
N,M = map(int,input().split())
for com in itertools.combinations_with_replacement(map(str,range(1,N+1)),M):
    print(' '.join(com))