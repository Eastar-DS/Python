#1 BFS python 시간초과 pypy메모리초과
import sys, collections
# input = sys.stdin.readline
N,M,K = map(int,input().split())
alpha = [input().rstrip() for _ in range(N)]
word = input().rstrip()
length = len(word)

dks = list(range(1,K+1))
queue = collections.deque([])
for i in range(N):
    for j in range(M):
        if alpha[i][j] == word[0]:
            queue.append([i,j,1])

output = 0
while(queue):
    x,y,i = queue.popleft()
    if i==length:
        output += 1
        continue
    for dk in dks:
        px,py,mx,my = x+dk,y+dk,x-dk,y-dk
        if px<N and alpha[px][y] == word[i]:
            queue.append([px,y,i+1])
        if py<M and alpha[x][py] == word[i]:
            queue.append([x,py,i+1])
        if mx>=0 and alpha[mx][y] == word[i]:
            queue.append([mx,y,i+1])
        if my>=0 and alpha[x][my] == word[i]:
            queue.append([x,my,i+1])
print(output)

#2 DFS
def dfs(x,y,i):
    global output
    if i==length:
        output += 1
        return
    for dk in range(1,K+1):
        px,py,mx,my = x+dk,y+dk,x-dk,y-dk
        if px<N and alpha[px][y] == word[i]:
            dfs(px,y,i+1)
        if py<M and alpha[x][py] == word[i]:
            dfs(x,py,i+1)
        if mx>=0 and alpha[mx][y] == word[i]:
            dfs(mx,y,i+1)
        if my>=0 and alpha[x][my] == word[i]:
            dfs(x,my,i+1)

import sys
# input = sys.stdin.readline

N,M,K = map(int,input().split())
alpha = [input().rstrip() for _ in range(N)]
word = input().rstrip()
length = len(word)

output = 0

for i in range(N):
    for j in range(M):
        if alpha[i][j] == word[0]:
            dfs(i,j,1)

print(output)