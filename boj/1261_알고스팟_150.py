#1 DFS 시간초과
def dfs(x,y,c):
    global output
    if x==N-1 and y==M-1:
        output = min(output,c)
        return
    if c>= output:
        return
    for k in range(4):
        i,j = x+dx[k],y+dy[k]
        if 0<=i<N and 0<=j<M and not visit[i][j]:
            visit[i][j] = 1
            if graph[i][j] == 1:
                dfs(i,j,c+1)
            else:
                dfs(i,j,c)
            visit[i][j] = 0
# import sys
# input = sys.stdin.readline
# sys.setrecursionlimit(15000)
M,N = map(int,input().split())
graph = [list(map(int,input().rstrip())) for _ in range(N)]
dx = [-1,1,0,0]
dy = [0,0,-1,1]
visit = [[0]*M for _ in range(N)]
visit[0][0] = 1

output = M*N
dfs(0,0,0)
print(output)



#2 BFS 안막혀있는곳부터 우선탐색하자.
M,N = map(int,input().split())
graph = [list(map(int,input())) for _ in range(N)]
dx = [-1,1,0,0]
dy = [0,0,-1,1]
visit = [[0]*M for _ in range(N)]
visit[0][0] = 1
output = M*N
import sys,collections
# input = sys.stdin.readline
queue = collections.deque([[0,0,0]])
while(queue):
    x,y,c = queue.popleft()
    if x==N-1 and y==M-1:
        print(c)
        break
    for k in range(4):
        i,j = x+dx[k],y+dy[k]
        if 0<=i<N and 0<=j<M and not visit[i][j]:
            visit[i][j] = 1
            if graph[i][j]:
                queue.append([i,j,c+1])
            else:
                queue.appendleft([i,j,c])

#왜 visit을 1로 만들어도 문제가 안생길까?
'''
01000
00010
11110 과같은 경우 0,2지점을 1로 처리하게 되는게 아닌가?
'''
'''
벽이 있는곳에서는 append를하고 벽이 없는곳에서 appendleft를 한다.
따라서 첫 c=0에서 이어지는 모든곳을 처리한 뒤 c=1을 처리하기시작하므로 문제가 없다.
'''

#3 queue에 x,y만 넣어보자. 더느리네.
M,N = map(int,input().split())
graph = [list(map(int,input())) for _ in range(N)]
dx = [-1,1,0,0]
dy = [0,0,-1,1]
visit = [[-1]*M for _ in range(N)]
visit[0][0] = 0
output = M*N
import sys,collections
# input = sys.stdin.readline
queue = collections.deque([[0,0]])
while(queue):
    x,y = queue.popleft()
    if x==N-1 and y==M-1:
        print(visit[x][y])
        break
    for k in range(4):
        i,j = x+dx[k],y+dy[k]
        if 0<=i<N and 0<=j<M and visit[i][j] == -1:
            if graph[i][j]:
                visit[i][j] = visit[x][y] + 1
                queue.append([i,j])
            else:
                visit[i][j] = visit[x][y]
                queue.appendleft([i,j])
