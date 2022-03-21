n,m = map(int,input().split())
output = 1
for _ in range(m):
    output *= n
    n-= 1
while m>0:
    output //=m
    m-=1
print(output)