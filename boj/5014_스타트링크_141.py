#1
import collections
F,S,G,U,D = map(int,input().split())
queue = collections.deque([[S,0]])
visit = [0]*(F+1)
visit[S] = 1
while(queue):
    now,c = queue.popleft()
    if now==G:
        print(c)
        break
    nextu,nextd = now+U,now-D
    if nextu <= F and not visit[nextu]:
        visit[nextu] = 1
        queue.append([nextu,c+1])
    if nextd > 0 and not visit[nextd]:
        visit[nextd] = 1
        queue.append([nextd,c+1])
else:
    print("use the stairs")
    
#2 더느리고 메모리도 많이잡아먹음.
import collections
F,S,G,U,D = map(int,input().split())
queue = collections.deque([S])
visit = [0]*(F+1)
visit[S] = 1
while(queue):
    now = queue.popleft()
    if now==G:
        print(visit[now]-1)
        break
    u,d = now+U,now-D
    if u <= F and not visit[u]:
        visit[u] = visit[now]+1
        queue.append(u)
    if d > 0 and not visit[d]:
        visit[d] = visit[now]+1
        queue.append(d)
else:
    print("use the stairs")