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