N,M = map(int,input().split())
nums = list(map(int,input().split()))
output = 0
summ, now = 0,0
for i in range(N):
    summ += nums[i]
    #summ이 M보다 크면 앞의 수들을 차례차례 빼준다.
    while(summ > M):
        summ -= nums[now]
        now += 1
    if summ == M:
        output+=1
print(output)