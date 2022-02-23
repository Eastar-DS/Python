#1 개느리네 ㅋㅋㅋ
import sys,itertools
input = sys.stdin.readline

N = int(input())
S = [list(map(int,input().split())) for _ in range(N)]
output = 10000
for orders in itertools.combinations(list(range(N)),N//2):
    nums = list(range(N))
    for order in orders:
        nums.remove(order)
    
    summ1,summ2 = 0,0
    for i in range(N//2-1):
        for j in range(i+1,N//2):
            summ1 += S[orders[i]][orders[j]]+S[orders[j]][orders[i]]
            summ2 += S[nums[i]][nums[j]]+S[nums[j]][nums[i]]
    
    output = min(output,abs(summ2-summ1))
print(output)

#2
