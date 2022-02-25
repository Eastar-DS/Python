n1,n2 = 1,1
for _ in range(int(input()) - 1):
    n1,n2 = n2,(2*n1+n2)%10007
print(n2)