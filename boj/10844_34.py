#1 DFS 시간초과 30만 넣어줘도 난리나네
N = int(input())
output = 0
def dfs(num,n):
    global output
    if len(num) == n:
        output += 1
        return
    new = int(num[-1])
    if new-1 >=0:
        dfs(num + str(new-1), N)
    if new+1 <10:
        dfs(num + str(new+1), N)
        
for i in range(1,10):
    dfs(str(i), N)

print(output % 1000000000)

#2 DP
N = int(input())
dp = [[0,1,1,1,1,1,1,1,1,1]]

for _ in range(N-1):
    before = dp[-1]
    new = [before[1]] + [before[i-1]+before[i+1] for i in range(1,9)] + [before[8]]
    dp.append(new)
    
print(sum(dp[-1]) % 1000000000)


# 0 1 2 3 4 5 6 7 8 9
# 0 1 1 1 1 1 1 1 1 1
# 1 1 2 2 2 2 2 2 2 1
# 1 3 3 4 4 4 4 4 3 2


#3
before = [0,1,1,1,1,1,1,1,1,1]
for _ in range(int(input()) - 1):
    before = [before[1]] + [before[i-1]+before[i+1] for i in range(1,9)] + [before[8]]
print(sum(before) % 1000000000)