#1
N = int(input())
M = int(input())
if M:
    broken = input().split()
else:
    broken = []
now = 100
output = abs(N-100)
for n in range(888889):
    for s in str(n):
        if s in broken:
            break
    else:
        output = min(output, len(str(n))+abs(n-N))
print(output)

#2
N = int(input())
M = int(input())
if not M:
    print(min(len(str(N)), abs(N-100)))
else:
    broken = input().split()
    output = abs(N-100)
    for n in range(888889):
        for s in str(n):
            if s in broken:
                break
        else:
            output = min(output, len(str(n))+abs(n-N))
    print(output)