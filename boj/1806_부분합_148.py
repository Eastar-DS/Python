#'2003_수들의합2'의 응용느낌
N,S = map(int,input().split())
nums = list(map(int,input().split()))
output,summ,now = N,0,0
#조건을 만족할 수 없으면 0출력
if sum(nums) < S:
    print(0)
    exit()
for i in range(N):
    #summ>=S 까지 쭉쭉 더해주기
    summ += nums[i]
    if summ < S:
        continue
    #summ>=S 상태에서 앞에서부터 빼주면서 최소길이 찾기
    while(summ>=S):
        summ -= nums[now]
        now += 1
    output = min(output,i-now+2)
print(output)