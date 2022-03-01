now = [1,1,1,1,1,1,1,1,1,1]
for _ in range(int(input()) - 1):
    for i in range(9):
        now[i+1] += now[i]
print(sum(now)%10007)

# 0 1 2 3 4 5 6 7 8 9
# 1 1 1 1 1 1 1 1 1 1
# 1 2 3 4 5 6 7 8 9 10