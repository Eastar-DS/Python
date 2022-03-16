#1
N = int(input())
P = [0] + list(map(int,input().split()))
dp = P[:]
for i in range(1,N+1):
    for j in range(1,i):
        tmp = dp[j]+P[i-j]
        if tmp > dp[i]:
            dp[i] = tmp
print(dp[-1])

#2 tmp를 리스트로바꾸니까 훨씬 빠르네.
N = int(input())
P = [0] + list(map(int,input().split()))
dp = P[:]
for i in range(2,N+1):
    dp[i] = max(dp[i], max([dp[j]+P[i-j] for j in range(1,i)]))
print(dp[-1])

#3 dp로만 답만들기
N = int(input())
dp = [0] + list(map(int,input().split()))
for i in range(2,N+1):
    dp[i] = max(dp[i], max([dp[j]+dp[i-j] for j in range(1,(i//2)+1)]))
print(dp[-1])