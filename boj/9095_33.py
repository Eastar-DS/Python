#1
for _ in range(int(input())):
    n1,n2,n3 = 1,2,4
    for __ in range(int(input()) - 1):
        n1,n2,n3 = n2,n3,n1+n2+n3
    print(n1)
