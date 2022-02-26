import sys
# input = sys.stdin.readline

#0: right 1:up 2:left 3: down
direction = 0
N = int(input())
graph = [[0]*N for _ in range(N)]
K = int(input())
for _ in range(K):
    x,y = map(int,input().split())
    graph[x-1][y-1] = 1

head = [0,0]
body = []
output,before = 0,0
for _ in range(int(input())):
    t, direc = input().split()
    if direc == 'D':
        direc = 3
    else:
        direc = 1
        
    for _ in range(int(t)-before):
        output += 1
        
        if direction == 0 :
            if (head[1] + 1) < N :
                if [head[0],head[1] + 1] in body:
                    print(output)
                    exit()
                if graph[head[0]][head[1] + 1] == 0:
                    body.insert(0,head)
                    body.pop()
                    head = [head[0],head[1] + 1]
                else:
                    body.insert(0,head)
                    head = [head[0],head[1] + 1]
                    graph[head[0]][head[1]] = 0
            else:
                print(output)
                exit()
        elif direction == 3 :
            if (head[0] + 1) < N :
                if [head[0] + 1,head[1]] in body:
                    print(output)
                    exit()
                if graph[head[0] + 1][head[1]] == 0:
                    body.insert(0,head)
                    body.pop()
                    head = [head[0] + 1,head[1]]
                else:
                    body.insert(0,head)
                    head = [head[0] + 1,head[1]]
                    graph[head[0]][head[1]] = 0
            else:
                print(output)
                exit()
        elif direction == 2 :
            if (head[1] - 1) >= 0 :
                if [head[0],head[1] - 1] in body:
                    print(output)
                    exit()
                if graph[head[0]][head[1] - 1] == 0:
                    body.insert(0,head)
                    body.pop()
                    head = [head[0],head[1] - 1]
                else:
                    body.insert(0,head)
                    head = [head[0],head[1] - 1]
                    graph[head[0]][head[1]] = 0
            else:
                print(output)
                exit()
        elif direction == 1 :
            if (head[0] - 1) >= 0 :
                if [head[0] - 1,head[1]] in body:
                    print(output)
                    exit()
                if graph[head[0] - 1][head[1]] == 0:
                    body.insert(0,head)
                    body.pop()
                    head = [head[0] - 1,head[1]]
                else:
                    body.insert(0,head)
                    head = [head[0] - 1,head[1]]
                    graph[head[0]][head[1]] = 0
            else:
                print(output)
                exit()
                
    before = int(t)
    direction = (direction + direc)%4

if direction == 0:
    for _ in range(N-head[1]):
        output += 1
        if (head[1] + 1) < N :
            if [head[0],head[1] + 1] in body:
                print(output)
                exit()
            if graph[head[0]][head[1] + 1] == 0:
                body.insert(0,head)
                body.pop()
                head = [head[0],head[1] + 1]
            else:
                body.insert(0,head)
                head = [head[0],head[1] + 1]
                graph[head[0]][head[1]] = 0
        else:
            print(output)
            exit()
elif direction == 3:
    for _ in range(N-head[0]):
        output += 1
        if (head[0] + 1) < N :
            if [head[0] + 1,head[1]] in body:
                print(output)
                exit()
            if graph[head[0] + 1][head[1]] == 0:
                body.insert(0,head)
                body.pop()
                head = [head[0] + 1,head[1]]
            else:
                body.insert(0,head)
                head = [head[0] + 1,head[1]]
                graph[head[0]][head[1]] = 0
        else:
            print(output)
            exit()
elif direction == 2: 
    for _ in range(head[1]+1):
        output += 1
        if (head[1] - 1) >= 0 :
            if [head[0],head[1] - 1] in body:
                print(output)
                exit()
            if graph[head[0]][head[1] - 1] == 0:
                body.insert(0,head)
                body.pop()
                head = [head[0],head[1] - 1]
            else:
                body.insert(0,head)
                head = [head[0],head[1] - 1]
                graph[head[0]][head[1]] = 0
        else:
            print(output)
            exit()
else:
    for _ in range(head[0] + 1):
        output += 1
        if (head[0] - 1) >= 0 :
            if [head[0] - 1,head[1]] in body:
                print(output)
                exit()
            if graph[head[0] - 1][head[1]] == 0:
                body.insert(0,head)
                body.pop()
                head = [head[0] - 1,head[1]]
            else:
                body.insert(0,head)
                head = [head[0] - 1,head[1]]
                graph[head[0]][head[1]] = 0
        else:
            print(output)
            exit()