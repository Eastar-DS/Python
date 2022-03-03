n = int(input())
g,dp = [],[0]*(n)
for _ in range(n):
    g.append(int(input()))
if n==1 or n==2:
    print(sum(g))
elif n==3:
    print(sum(g)-min(g))
else:
    dp[0],dp[1] = g[0],g[0]+g[1]
    dp[2] = g[0]+g[1]+g[2]-min(g[0],g[1],g[2])
    for i in range(3,n):
        dp[i] = max(dp[i-3]+g[i-1]+g[i], dp[i-2]+g[i], dp[i-1])
    print(dp[-1])


# [6,10,13,9,8,1]
# 6 16 23

#[0, 0, 10, 0, 5, 10, 0, 0, 1, 10, 0, 0]
# oxoxooxxooxx

#왜 xx가 있어야하는지를 생각해내야함.
# oox(x)oo