# - 0 +로 구분해서 - +로 만들고 0있으면 *2
#1 combinations, 0예외처리하면 약간더 빨라짐 396ms
import itertools
N,S = map(int,input().split())
nums = list(map(int,input().split()))
zero = 0
if 0 in nums:
    nums.remove(0)
    N -= 1
    zero = 1
output = 0
for i in range(1,N+1):
    for com in itertools.combinations(nums,i):
        if sum(com) == S:
            output += 1

if zero:
    if S==0:
        output = output*2 +1
    else:
        output *= 2
print(output)




#2 DFS 532ms
def dfs(idx, sum):
    global cnt
    if idx >= n:
        return
    sum += s_[idx]
    if sum == s:
        cnt += 1
    dfs(idx + 1, sum - s_[idx])
    dfs(idx + 1, sum)
n, s = map(int, input().split())
s_ = list(map(int, input().split()))
cnt = 0
dfs(0, 0)
print(cnt)




#3 반으로 쪼개기 젤빠 68ms
def dfs(index,end,dic,summ):
    global output
    if index == end:
        return
    dfs(index+1,end,dic,summ)
    summ += nums[index]
    dic[summ] = dic.get(summ,0) + 1
    dfs(index+1,end,dic,summ)

N,S = map(int,input().split())
nums = list(map(int,input().split()))
output = 0
left = nums[:N//2]
right = nums[N//2:]
dicl,dicr = {},{}

dfs(0,N//2,dicl,0)
dfs(N//2,N,dicr,0)
output = dicl.get(S,0)+dicr.get(S,0)
for summ in dicl:
    output += dicl[summ]*dicr.get(S-summ,0)
print(output)




#4 #3에 zero까지 추가해보자. 68ms똑같은 속도로 나오네... 
#반례 0가득하게 주어지면 이게 더 좋을듯!
def dfs(index,end,dic,summ):
    global output
    if index == end:
        return
    #현재 index값을 포함하지 않은것 실행
    dfs(index+1,end,dic,summ)
    summ += nums[index]
    dic[summ] = dic.get(summ,0) + 1
    if summ == S:
        output += 1    
    dfs(index+1,end,dic,summ)

N,S = map(int,input().split())
nums = list(map(int,input().split()))
output = 0
zero = 0
while 0 in nums:
    zero += 1
    nums.remove(0)
    N-=1

left = nums[:N//2]
right = nums[N//2:]
dicl,dicr = {},{}
dfs(0,N//2,dicl,0)
dfs(N//2,N,dicr,0)
for summ in dicl:
    output += dicl[summ]*dicr.get(S-summ,0)

if zero:
    if S==0:
        output = output*(2**zero) + (2**zero-1)
    else:
        output *= (2**zero)
print(output)




#5 1번을 반으로 쪼개서 뭐가문젠지 모르겠다. 안됨.
import itertools
N,S = map(int,input().split())
nums = list(map(int,input().split()))
output = 0
d1,d2 = {},{}
l,r = nums[:N//2], nums[N//2:]
for i in range(1,N//2):
    for com in itertools.combinations(l,i):
        summ = sum(com)
        d1[summ] = d1.get(summ,0) + 1
for i in range(1,len(r)):
    for com in itertools.combinations(r,i):
        summ = sum(com)
        d2[summ] = d2.get(summ,0) + 1
output = d1.get(S,0)+d2.get(S,0)
for summ in d1:
    output += d1[summ]*d2.get(S-summ,0)
print(output)