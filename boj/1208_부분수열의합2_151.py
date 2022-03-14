def dfs(index,end,dic,summ):
    if index == end:
        return
    
    dfs(index+1,end,dic,summ)
    summ += nums[index]
    dic[summ] = dic.get(summ,0) + 1
    dfs(index+1,end,dic,summ)    

N,S = map(int,input().split())
nums = list(map(int,input().split()))
l,r = nums[:N//2], nums[N//2:]
dl,dr = {},{}

dfs(0,N//2,dl,0)
dfs(N//2,N,dr,0)
output = dl.get(S,0)+dr.get(S,0)
for s in dl:
    output += dl[s]*dr.get(S-s,0)
print(output)