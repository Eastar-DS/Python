#1 DFS 시간초과
def dfs(index,now,summ):
    global output
    if index == V:
        if summ < output:
            output = summ
        return
    for v,c in graph[now]:
        if not visit[v]:
            visit[v] = 1
            dfs(index+1, v, summ+c)
            visit[v] = 0

import sys
sys.setrecursionlimit(20000)
# input = sys.stdin.readline
V,E = map(int,input().split())
graph = [[] for _ in range(V+1)]
for _ in range(E):
    a,b,c = map(int,input().split())
    graph[a].append([b,c])
    graph[b].append([a,c])

output = 2147483649
visit = [0]*(V+1)
for i in range(1,V):
    visit[i] = 1
    dfs(1,i,0)
    visit[i] = 0
print(output)