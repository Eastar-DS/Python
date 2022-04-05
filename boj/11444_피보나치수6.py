a,b = 0,1
for _ in range(int(input()) - 1):
    a,b = b,a+b
    if b >= 1000000007:
        b %= 1000000007
print(b)