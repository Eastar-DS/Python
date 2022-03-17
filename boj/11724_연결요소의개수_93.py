#1 BFS
import sys,collections
# input = sys.stdin.readline

N,M = map(int,input().split())
graph = [[] for _ in range(N+1)]
for _ in range(M):
    a,b = map(int,input().split())
    graph[a].append(b)
    graph[b].append(a)

output = 0
visit = [0]*(N+1)
for i in range(1,N+1):
    if not visit[i]:
        output += 1
        queue = collections.deque([i])
        while(queue):
            now = queue.popleft()
            if visit[now]:
                continue
            visit[now] = 1
            if graph[now]:
                for num in graph[now]:
                    queue.append(num)
            graph[now] = []
print(output)

#2 DFS
def dfs(n):
    visit[n] = 1
    for i in graph[n]:
        if not visit[i]:
            dfs(i)
    
import sys
# input = sys.stdin.readline
sys.setrecursionlimit(1001)
N,M = map(int,input().split())
graph = [[] for _ in range(N+1)]
for _ in range(M):
    a,b = map(int,input().split())
    graph[a].append(b)
    graph[b].append(a)

output = 0
visit = [0]*(N+1)
for i in range(1,N+1):
    if not visit[i]:
        output += 1
        dfs(i)
print(output)
