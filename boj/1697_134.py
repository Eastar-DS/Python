#1 BFS
from collections import deque
N,M = map(int,input().split())
queue = deque([[N,0]])
visit = [0]*100001
while(queue):
    n,c = queue.popleft()
    if n==M:
        print(c)
        break
    if n-1>=0 and not visit[n-1]:
        queue.append([n-1,c+1])
        visit[n-1] = 1
    if n+1<=100000 and not visit[n+1]:
        queue.append([n+1,c+1])
        visit[n+1] = 1
    if 2*n<=100000 and not visit[n*2]:
        queue.append([n*2,c+1])
        visit[n*2] = 1
        
#2 와...
def find(n, k):
    if n >= k:
        return n-k
    elif k == 1:
        return 1
    elif k%2:
        return min(find(n, k-1), find(n, k+1)) + 1
    else:
        return min(k-n, find(n, k//2) + 1)
  
import sys
n, k = map(int, sys.stdin.readline().split())
print(find(n, k))