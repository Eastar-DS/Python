#sum(visit)을 안쓰고 summ을 만들어서 쓰니까 통과.
def dfs(x,y):
    global output,summ
    if visit[graph[x][y]] :
        output = max(output,summ)
        return    
    if output == maximum:
        print(output)
        exit()
    visit[graph[x][y]] = 1
    summ += 1
    for k in range(4):
        i,j = x+dx[k],y+dy[k]
        if 0<=i<R and 0<=j<C:
            dfs(i,j)
    visit[graph[x][y]] = 0
    summ -= 1
import sys
# input = sys.stdin.readline
R,C = map(int,input().split())
graph = [list(map(lambda x: ord(x)-65,input().rstrip())) for _ in range(R)]
visit,summ = [0]*26,0
dx = [-1,1,0,0]
dy = [0,0,-1,1]
output = 0
#maximum을 그래프한번 순회하면서 잡아두기 100ms빨라짐
maximum = [0]*26
for i in range(R):
    for j in range(C):
        maximum[graph[i][j]]=1
maximum = sum(maximum)
dfs(0,0)
print(output)

#2 좀더빠르다. 이코드에선 maximum안구하는게 더빠르네. dfs(x,y,summ)이 더느림.
def dfs(x,y):
    global output,summ
    if output < summ:
        output = summ
    for k in range(4):
        i,j = x+dx[k],y+dy[k]
        if 0<=i<R and 0<=j<C and not visit[graph[i][j]]:
            visit[graph[i][j]] = 1
            summ += 1
            dfs(i,j)
            visit[graph[i][j]] = 0
            summ -= 1
import sys
# input = sys.stdin.readline
R,C = map(int,input().split())
graph = [list(map(lambda x: ord(x)-65,input().rstrip())) for _ in range(R)]
visit,summ = [0]*26,1
visit[graph[0][0]] = 1
dx = [-1,1,0,0]
dy = [0,0,-1,1]
output = 0
dfs(0,0)
print(output)

