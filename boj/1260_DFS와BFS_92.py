#1
def dfs(n):
    if visit[n] :
        return
    visit[n] = 1
    d.append(str(n))
    #간선이 0인경우 keyerror발생.
    for node in graph.get(n,[]):
        dfs(node)

def bfs(n):
    queue = collections.deque([n])
    while queue:
        now = queue.popleft()
        if visit[now]:
            continue
        visit[now] = 1
        b.append(str(now))
        #간선이 0인경우 keyerror발생.
        for node in graph.get(now,[]):
            queue.append(node)

import sys,collections
# input = sys.stdin.readline
N,M,V = map(int,input().split())
graph = {}
for _ in range(M):
    a,b = map(int,input().split())
    graph[a] = graph.get(a,[]) + [b]
    graph[b] = graph.get(b,[]) + [a]
for key in graph:
    graph[key].sort()

d,b=[],[]
visit=[0]*(N+1)
dfs(V)
visit=[0]*(N+1)
bfs(V)
print(' '.join(d))
print(' '.join(b))

#2 속도를 높여보자.
def dfs(n):
    visit[n] = 1
    d.append(str(n))
    if n in graph:
        for node in graph[n]:
            if not visit[node]:
                dfs(node)

def bfs(V):
    queue = collections.deque([V])
    while queue:
        now = queue.popleft()
        b.append(str(now))
        if now in graph:
            for node in graph[now]:
                if not visit[node]:
                    queue.append(node)
                    visit[node] = 1
import sys,collections
# input = sys.stdin.readline
N,M,V = map(int,input().split())
graph = {}
for _ in range(M):
    a,b = map(int,input().split())
    if a in graph:
        graph[a].append(b)
    else:
        graph[a] = [b]
    if b in graph:
        graph[b].append(a)
    else:
        graph[b] = [a]
for key in graph:
    graph[key].sort()

d,b=[],[]
visit=[0]*(N+1)
dfs(V)
visit=[0]*(N+1)
visit[V] = 1
bfs(V)
print(' '.join(d))
print(' '.join(b))


#3 graph를 list로
def dfs(n):
    visit[n] = 1
    d.append(str(n))
    for node in graph[n]:
        if not visit[node]:
            dfs(node)        

def bfs(V):
    queue = collections.deque([V])
    while queue:
        now = queue.popleft()
        b.append(str(now))
        for node in graph[now]:
            if not visit[node]:
                queue.append(node)
                visit[node] = 1
import sys,collections
# input = sys.stdin.readline
N,M,V = map(int,input().split())
graph = [[] for _ in range(N+1)]
for _ in range(M):
    a,b = map(int,input().split())
    graph[a].append(b)
    graph[b].append(a)
for i in range(1,N+1):
    graph[i].sort()

d,b=[],[]
visit=[0]*(N+1)
dfs(V)
visit=[0]*(N+1)
visit[V] = 1
bfs(V)
print(' '.join(d))
print(' '.join(b))