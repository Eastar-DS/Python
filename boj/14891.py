import sys
input = sys.stdin.readline

def clock(index,direction,dilist):
    for di in dilist:
        if di == 1 and index+1<=4 and clocks[index][2] != clocks[index+1][6]:
            clock(index+1, -direction, [1])
        elif di == -1 and index-1>=1 and clocks[index][6] != clocks[index-1][2]:
            clock(index-1, -direction, [-1])
    if direction == 1:
        clocks[index] = [clocks[index][-1]] + clocks[index][:-1]
    else:
        clocks[index] = clocks[index][1:] + [clocks[index][0]]

clocks = [0]+[list(input().rstrip()) for _ in range(4)]
for _ in range(int(input())):
    index, direction = map(int,input().split())
    clock(index,direction,[-1,1])

print(sum([2**i*int(clocks[i+1][0]) for i in range(4)]))