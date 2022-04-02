#다각형의 넓이
# import sys
# input = sys.stdin.readline
N = int(input())
x1,y1 = map(int,input().split())
lx,ly = x1,y1
output = 0
for _ in range(N-1):
    x2,y2 = map(int,input().split())
    output += x1*y2-x2*y1
    x1,y1 = x2,y2
output = abs(output+x1*ly-lx*y1)/2
print(f'{output : .1f}'.lstrip())