#와 랭킹3위!
import sys
# input = sys.stdin.readline

N = int(input())
isfull = [[0]*N for _ in range(N)]
seats = {}
dic = {}
dx,dy = [-1,1,0,0],[0,0,-1,1]
def manyEmpty(x,y):
    output = 5
    for t in range(4):
        i,j = x+dx[t], y+dy[t]
        if 0<=i<N and 0<=j<N:
            output -= isfull[i][j]
        else:
            output -= 1
    return output

def noFriend():
    k = 0
    for x in range(N):
        for y in range(N):
            if not isfull[x][y]:
                tmp = manyEmpty(x,y)
                if tmp > k:
                    i,j,k = x,y,tmp
                if k == 5:
                    return [i,j]
    return [i,j]
for _ in range(N**2):
    tmp ={}
    student, *friends = input().rstrip().split()
    dic[student] = friends
    #친구많은칸 찾기
    for friend in friends:
        if friend in seats:
            x,y = seats[friend]
            for t in range(4):
                i,j = x+dx[t], y+dy[t]
                if 0<=i<N and 0<=j<N and not isfull[i][j]:
                    tmp[(i,j)] = tmp.get((i,j),0) + 1
    f = [[item[0][0], item[0][1], item[1]] for item in list(tmp.items())]
    f.sort(key=lambda x : (x[2], manyEmpty(x[0],x[1]), -x[0], -x[1]))
    #자리정하기
    i,j = -1,-1
    if f:
        i,j,__ = f[-1]
    else:
        i,j = noFriend()                

    isfull[i][j] = 1
    seats[student] = (i,j)

#답구하기
output = 0
def isFriend(s):
    c = 0
    x,y = seats[s]
    friends = dic[s]
    for friend in friends:
        i,j = seats[friend]
        if abs(x-i) + abs(y-j) == 1:
            c += 1
    if c==0:
        return 0
    else:
        return 10**(c-1)        
            
for x in range(1,N**2+1):
    output += isFriend(str(x))

print(output)


# 3
# 7 9 3 8 2 
# 5 7 3 8 6
# 3 5 2 4 9
# 9 6 8 3 4
# 8 5 3 1 6
# 6 3 8 5 4
# 2 6 4 8 7
# 1 8 3 4 5
# 4 7 9 3 8

# 정답 : 151
# 비고 :
# 3 5 8
# 9 7 6
# 1 2 4