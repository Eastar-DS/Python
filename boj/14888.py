#1 permutation
import itertools
N = int(input())
nums = list(map(int,input().split()))
oper = list(map(int,input().split()))
minimum,maximum = 10**9,-10**9

for order in set(itertools.permutations([0]*oper[0]+[1]*oper[1]+[2]*oper[2]+[3]*oper[3],N-1)):
    summ = nums[0]
    for i,o in enumerate(order,1):
        if o==0:
            summ += nums[i]
        elif o==1:
            summ -= nums[i]
        elif o==2:
            summ *= nums[i]
        else:
            if summ < 0:
                summ = -((-summ) // nums[i])
            else:
                summ //= nums[i]
    minimum = min(minimum,summ)
    maximum = max(maximum,summ)
print(maximum)
print(minimum)

#2 dfs
N = int(input())
nums = list(map(int,input().split()))
oper = list(map(int,input().split()))
minimum,maximum = 10**9,-10**9

def dfs(i,summ,p,m,mul,div):
    global minimum,maximum
    if i==N:
        minimum = min(minimum,summ)
        maximum = max(maximum,summ)
        return
    if p:
        dfs(i+1,summ+nums[i],p-1,m,mul,div)
    if m:
        dfs(i+1,summ-nums[i],p,m-1,mul,div)
    if mul:
        dfs(i+1,summ*nums[i],p,m,mul-1,div)
    if div:
        dfs(i+1,int(summ/nums[i]),p,m,mul,div-1)
        # if summ < 0:
        #     dfs(i+1,-((-summ)//nums[i]),p,m,mul,div-1)
        # else:
        #     dfs(i+1,summ//nums[i],p,m,mul,div-1)
    
dfs(1,nums[0],oper[0],oper[1],oper[2],oper[3])
print(maximum)
print(minimum)