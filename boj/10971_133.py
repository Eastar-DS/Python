N = int(input())
W = [list(map(int,input().split())) for _ in range(N)]

output = 10000000
def dfs(i,visit,start,out):
    global output
    #합을구하다 기존값보다 커지면 탐색종료
    if out >= output:
        return
    #탐색 다했으면 비교할당
    if len(visit) == N+1:
        output = min(output,out)
    #마지막에서 시작점
    if len(visit) == N and W[i][start]:
        dfs(start,visit+[start],start,out+W[i][start])
    #방문하지 않은 노드탐색
    for j in range(N):
        if j not in visit and W[i][j]:
            dfs(j, visit+[j], start, out+W[i][j])

for i in range(N):
    dfs(i,[i],i,0)
print(output)

#visit을 밖으로 빼보자. 같은메모리, 같은속도
N = int(input())
W = [list(map(int,input().split())) for _ in range(N)]

output = 10000000
def dfs(i,start,out):
    global output
    #합을구하다 기존값보다 커지면 탐색종료
    if out >= output:
        return
    #탐색 다했으면 비교할당
    if len(visit) == N+1:
        output = min(output,out)
    #마지막에서 시작점
    if len(visit) == N and W[i][start]:
        visit.append(start)
        dfs(start,start,out+W[i][start])
        visit.pop()
    #방문하지 않은 노드탐색
    for j in range(N):
        if j not in visit and W[i][j]:
            visit.append(j)
            dfs(j,start,out+W[i][j])
            visit.pop()

for i in range(N):
    visit = [i]
    dfs(i,i,0)
print(output)