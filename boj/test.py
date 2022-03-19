#2011


#2133
N = int(input())
if N%2==1:
    print(0)
else:
    N //= 2
    dp = [3] + [2]*(N-1)
    summ = 0
    for i in range(1,N):
        dp[i] += 3*dp[i-1]+summ*2
        summ += dp[i-1]
    print(dp[-1])
#11053
N=int(input())
nums = list(map(int,input().split()))
dp=[1]*N
for i in range(1,N):
    for j in range(i):
        if nums[j] < nums[i] and dp[j]+1 > dp[i]:
            dp[i] = dp[j]+1
print(max(dp))

#2156
n = int(input())
nums = [0,0] + [int(input()) for _ in range(n)]
if n==1 or n==2:
    print(sum(nums))
else:
    dp = [0,0,nums[2],nums[2]+nums[3]]+[0]*(n-2)
    for i in range(4,n+2):
        dp[i] = max(dp[i-4]+nums[i-1]+nums[i], dp[i-3]+nums[i-1]+nums[i], dp[i-2]+nums[i], dp[i-1])
    print(dp[-1])

# o x x o o
# o x o o x
# o x o x o
# o o x o o
# n n-4 n-3 n-2 max(n-1) 


#1182
def makedic(now,end,dic,ns,summ):
    if now==end:
        return
    makedic(now+1,end,dic,ns,summ)
    summ += ns[now]
    dic[summ] = dic.get(summ,0)+1
    makedic(now+1,end,dic,ns,summ)
N,S = map(int,input().split())
nums = list(map(int,input().split()))
l,r = nums[:N//2], nums[N//2:]
ldic,rdic,lend,rend = {},{},N//2,N-N//2
makedic(0,lend,ldic,l,0)
makedic(0,rend,rdic,r,0)
output = ldic.get(S,0)+rdic.get(S,0)
for s in ldic:
    output += ldic[s]*rdic.get(S-s,0)
print(output)

#2580
def makenums(i,j):
    nums = ['1','2','3','4','5','6','7','8','9']
    row = graph[i]
    for num in row:
        if num in nums:
            nums.remove(num)
    col = [graph[x][j] for x in range(9)]
    for num in col:
        if num in nums:
            nums.remove(num)
    x,y = (i//3)*3,(j//3)*3
    box = [graph[a][b] for a in [x,x+1,x+2] for b in [y,y+1,y+2]]
    for num in box:
        if num in nums:
            nums.remove(num)
    return nums
def dfs(now):
    if now == end:
        for line in graph:
            print(' '.join(line))
        exit()
    i,j = zeros[now]
    nums = makenums(i,j)
    for num in nums:
        graph[i][j] = num
        dfs(now+1)
        graph[i][j] = '0'
graph = [input().split() for _ in range(9)]
zeros = []
for i in range(9):
    for j in range(9):
        if graph[i][j] == '0':
            zeros.append([i,j])
end = len(zeros)
dfs(0)

#1759
from itertools import combinations
L,C = map(int,input().split())
alphas = sorted(input().split())
for com in combinations(alphas,L):
    aeiou,alpha = 0,0
    for s in com:
        if s in 'aeiou':
            aeiou += 1
        else:
            alpha += 1
    if aeiou and alpha>=2:
        print(''.join(com))


#2186 word 거꾸로가면서 만들어보자. 오졌다.

import sys,collections
# input = sys.stdin.readline

N,M,K = map(int,input().split())
graph = [list(input()) for _ in range(N)]
word = input().rstrip()
length = len(word)
visit = [[[0]*length for _ in range(M)] for _ in range(N)]

queue = collections.deque([])
output = 0
for i in range(N):
    for j in range(M):
        if graph[i][j] == word[-1]:
            queue.append([i,j,length-1])
            visit[i][j][length-1] = 1
while(queue):
    i,j,index = queue.popleft()
    if not index:
        output += visit[i][j][index]
        continue
    for k in range(1,K+1):
        if 0<=i-k and graph[i-k][j]==word[index-1]:
            if not visit[i-k][j][index-1]:
                queue.append([i-k,j,index-1])
            visit[i-k][j][index-1] += visit[i][j][index]
        if 0<=j-k and graph[i][j-k]==word[index-1]:
            if not visit[i][j-k][index-1]:
                queue.append([i,j-k,index-1])
            visit[i][j-k][index-1] += visit[i][j][index]
        if i+k<N and graph[i+k][j]==word[index-1]:
            if not visit[i+k][j][index-1]:
                queue.append([i+k,j,index-1])
            visit[i+k][j][index-1] += visit[i][j][index]
        if j+k<M and graph[i][j+k]==word[index-1]:
            if not visit[i][j+k][index-1]:
                queue.append([i,j+k,index-1])
            visit[i][j+k][index-1] += visit[i][j][index]

print(output)
