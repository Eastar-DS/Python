#A
import sys
input = sys.stdin.readline

f,w = 'for','while'
output = 0
for _ in range( int(input()) ):
    string = input()
    tmp = 0
    for i in range(len(string) - 2):
        if string[i:i+3] == f or string[i:i+5] == w:
            tmp += 1
    output = max(output,tmp)
print(output)



#B
# 국영수과
# 동점은 번호가 빠른사람이
import sys
input = sys.stdin.readline

K,E,M,S = [],[],[],[]
for _ in range(int(input())):
    i,k,e,m,s = map(int,input().split())
    K.append([i,k])
    E.append([i,e])
    M.append([i,m])
    S.append([i,s])

K.sort(key = lambda x : (-x[1],x[0]))
E.sort(key = lambda x : (-x[1],x[0]))
M.sort(key = lambda x : (-x[1],x[0]))
S.sort(key = lambda x : (-x[1],x[0]))

output = [K[0][0]]
for index,score in E:
    if index in output:
        continue
    output.append(index)
    break

for index,score in M:
    if index in output:
        continue
    output.append(index)
    break

for index,score in S:
    if index in output:
        continue
    output.append(index)
    break

print(' '.join([str(index) for index in output]))



#C
#queue = 0 stack = 1
import sys
input = sys.stdin.readline

N = int(input())
shape = list(map(int,input().split()))
nums = list(map(int,input().split()))
nums = [nums[i] for i in range(N) if shape[i] == 0]
M = int(input())
C = list(map(int,input().split()))

print(' '.join([str(num) for num in (nums[::-1] + C)[:M]]))





#E pypy제출
import sys, collections
input = sys.stdin.readline
    
N,M = map(int,input().split())
queue = collections.deque()
graph = []
for i in range(N):
    line = input().split()
    graph.append(line)
    for j in range(M):
        if line[j] == '1':
            queue.append([i,j,1,0])
            graph[i][j] = [1,0]
        elif line[j] == '2':
            queue.append([i,j,2,0])
            graph[i][j] = [2,0]

dx,dy = [-1,1,0,0],[0,0,-1,1]
out1,out2,out3 = 1,1,0

def disease(x,y,g,c):
    global out1,out2,out3
    for t in range(4):
        i,j = x+dx[t], y+dy[t]
        if(0<=i<N) and (0<=j<M):
            if graph[i][j] == '0':
                graph[i][j] = [g,c+1]
                queue.append([i,j,g,c+1])
                if g==1:
                    out1 += 1
                else:
                    out2 += 1
            elif graph[i][j] in ['-1','3']:
                continue
            elif graph[i][j][0] == g:
                continue
            elif graph[i][j][1] == c+1:
                graph[i][j] = '3'
                out3 += 1
                if g==1:
                    out2 -= 1
                else:
                    out1 -= 1

while(queue):
    i,j,g,c = queue.popleft()
    if graph[i][j] == '3':
        continue
    disease(i,j,g,c)
print(str(out1) + ' ' + str(out2) + ' ' + str(out3))


# queue = collections.deque([[1,3,2,0],[3,7,1,0]])
# graph = [['0', '0', '0', '0', '0', '0', '0', '0', '0'],
#  ['0', '0', '0', [2,0], '0', '0', '-1', '0', '0'],
#  ['0', '0', '0', '0', '0', '0', '0', '0', '0'],
#  ['0', '0', '0', '-1', '0', '0', '0', [1,0], '0'],
#  ['0', '0', '0', '0', '0', '0', '0', '0', '0'],
#  ['0', '0', '0', '0', '0', '0', '-1', '0', '0'],
#  ['0', '0', '0', '0', '0', '0', '0', '0', '0']]




#F
import sys
input= sys.stdin.readline

#K : 진법, N: N번째 숫자 출력
#XK(N) : K진법으로 1~N까지 나열한수
#N이 주어지면 K진법의 어떤수를 나타내는지 알아야 할것같다.
T,K = map(int,input().split())
for _ in range(T):
    N = int(input())
    length,d = 0,1
    while(length < N):
        length += d * (k-1) * k**(d-1)
    




# 1~k-1 : 1자리
# k~k**2-1 : 2자리
# k**2 ~ k**3-1 : 3자리

# 따라서 1자리가 k - 1개, 2자리가 k**2 - k개, 3자리가 k**3 - k**2개 이다.
# 1~k-1, (k-1)+2 ~ (k-1)+2*(k**2 - k),




#G 로미오가 최소 몇개의 땅을 지나가야 하는지를 세면 될거같다. 아니네 ㄷㄷ;
import sys, collections
input = sys.stdin.readline

graph = []
N = int(input())
for _ in range(N):
    graph.append(list(input().rstrip()))

graph[0][0], graph[-1][-1] = 0,0

dx,dy = [-1,1,0,0],[0,0,-1,1]
def makeGroup(x,y,c):
    now = graph[x][y]
    graph[x][y] = c
    for t in range(4):
        i,j = x+dx[t], y+dy[t]
        if(0<=i<N) and (0<=j<N) and graph[i][j] == now:
            makeGroup(i,j,c)
        

g = 1
for i in range(N):
    for j in range(N):
        if type(graph[i][j]) == str:
            makeGroup(i,j,g)
            g+=1

visited,output = [0] * (g+1), g
def dfs(x,y,c):
    global output
    if x == N-1 and y == N-1:
        output = min(output,c)
        return
    now = graph[x][y]
    visited[graph[x][y]] = 1
    for t in range(4):
        i,j = x+dx[t], y+dy[t]
        if(0<=i<N) and (0<=j<N) :
            if graph[i][j] == now:
                queue.appendleft([i,j,c])
            else:
                queue.append([i,j,c+1])







queue = collections.deque([[0,0,0]])
while queue:
    x,y,c = queue.popleft()
    now = graph[x][y]
    if x == N-1 and y == N-1:
        print(c)
        break
    for t in range(4):
        i,j = x+dx[t], y+dy[t]
        if(0<=i<N) and (0<=j<N) :
            if graph[i][j] == now:
                queue.appendleft([i,j,c])
            else:
                queue.append([i,j,c+1])
    






graph = [[0, 1, 1, 1, 2, 2],
 [3, 1, 1, 2, 2, 4],
 [3, 1, 5, 6, 7, 4],
 [3, 1, 8, 6, 7, 4],
 [3, 9, 8, 7, 7, 4],
 [9, 9, 8, 7, 7, 0]]




#H 와시바 걍 답이 이거였음
print(' '.join([str(2*s + 1) for s in range(int(input()))]))

# import time


# N = int(input())
# S = time.time()
# if N==1 :
#     print(5)
# output = [1]
# visited = [0]*100001
# visited[1] = 1

# #start+1 자리부터 2자리까지 나누어 떨어지는지 확인하기
# def isGood(num,start,summ):
#     d = 0
#     while(start > 1):
#         if summ % start == 0:
#             start -=1
#             summ -= output[d]
#             d += 1
#         else:
#             return False
#     return True
    
    

# start,summ = 1,1
# while(start < N):
#     for num in range(start+1 - summ%(start+1), 100001, start+1):
#     # for num in range(100000 - 100000%(start+1) - summ%(start+1),0, -(start+1)):
#         if visited[num]:
#             continue
#         if isGood(num,start+1,summ+num):
#             visited[num] = 1
#             start += 1
#             summ += num
#             output.append(num)
#             break

# print(' '.join([str(s) for s in output]))

# print(time.time() - S)




#I
N,M = map(int,input().split())
output = 0
for i in range(1,N+1):
    output += (N//i) * (i%M)
    if output > 1000000006:
        output %= 1000000007
print(output)

# 4 3

# 4//1 * 1%3 = 4
# 4//2 * 2%3 = 4
# 4//3 * 3%3 = 0
# 4//4 * 4%3 = 1


# N = N//x * x + N%x

# x = x//M * M + x%M

# N = N//x * (x//M * M + x%M) + N%x

#   = N//x * x%M + N//x * x//M * M + N%x

# N//x * x%M = N//x * x - N//x * x//M * M 

#  = (N-N%x) - (N//x * (x-x%M))

#  = (N-N%x) - ((N-N%x) * (x-x%M)/x)




#J
for _ in range(int(input())):
    A,B,K = map(int,input().split())
    summ,start = K*(K+1)/2,'swoon'
    if (K%2) == 1 and ((B-A)//summ) % 2 == 1:
        start = 'raararaara'
    B = (B-A)%summ
    if B in [K+1, 2*K+1, 3*K+1, 4*K+1, 5*K+1]:
        if start == 'swoon':
            start = 'raararaara'
        else :
            start = 'swoon'
    print(start)
    



'''
1~K의 합으로 B이상이 만들어지면 여기서 먼저 시작하는놈이 이기나? 아니네
A B K
2 4 7

4 10 5

B가 10이하면 무조건 선승
B가 11이면 후승 (11 을 갖는자가 승리)
B가 12~20 이면 선승
B가 21이면 후승 (21 을 갖는자가 승리)
B가 22~29 이면 선승
B가 30,31

1 29
8 21
2 19
7 12


'''











