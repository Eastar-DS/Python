#1 삼각형
#nums[i]가 nums[j]들이랑 안이어져있는지 확인하고 graph[i]에서 n을 제거, 한바퀴돌리고 graph[n]을 비우기
def connected(n):
    nums,length = graph[n],len(graph[n])
    for i in range(length-1):
        graph[nums[i]].remove(n)
        for j in range(1,length):
            if nums[i] in graph[nums[j]]:
                return 1
    graph[n] = []
    return 0
import sys
# input = sys.stdin.readline
for _ in range(int(input())):
    V,E = map(int,input().split())
    graph = [[] for _ in range(V+1)]
    for _ in range(E):
        a,b = map(int,input().split())
        graph[a].append(b)
        graph[b].append(a)

    for i in range(1,V+1):
        if len(graph[i])>=2 and connected(i):
            print("NO")
            break
    else:
        print("YES")

#2 BFS 1672ms
def bfs(i):
    global group
    queue = collections.deque([[i]])
    while queue:
        nums = queue.popleft()
        next_nums = []
        for num in nums:
            if visit[num] == -group:
                return 1
            elif not visit[num]:
                visit[num] = group
                next_nums += graph[num]
        if next_nums:
            queue.append(set(next_nums))
            group *= -1
    return 0
import sys,collections
# input = sys.stdin.readline
for _ in range(int(input())):
    V,E = map(int,input().split())
    graph = [[] for _ in range(V+1)]
    for _ in range(E):
        a,b = map(int,input().split())
        graph[a].append(b)
        graph[b].append(a)
    
    visit = [0]*(V+1)
    group = 1
    for i in range(1,V+1):
        if not visit[i]:
            if bfs(i):
                print("NO")
                break
    else:
        print("YES")
                        
#3 DFS 1368ms set22222
def dfs(i,group):
    global check
    if check:
        return
    visit[i] = group
    for num in graph[i]:
        if visit[num] == group:
            check = 1
            return 
        elif not visit[num]:
            dfs(num,-group)

import sys,collections
# input = sys.stdin.readline
sys.setrecursionlimit(20001)
for _ in range(int(input())):
    V,E = map(int,input().split())
    graph = [[] for _ in range(V+1)]
    for _ in range(E):
        a,b = map(int,input().split())
        graph[a].append(b)
        graph[b].append(a)
    
    visit = [0]*(V+1)
    check = 0
    for i in range(1,V+1):
        if not visit[i]:
            dfs(i,1)
            if check:
                print("NO")
                break
    else:
        print("YES")





