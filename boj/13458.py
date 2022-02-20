N = int(input())
As = list(map(int, input().split()))
B,C = map(int,input().split())

for A in As:
    if(A-B>0):
        N += ((A-B - 1)//C + 1)

print(N)