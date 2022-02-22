#A
#기준과 양쪽수중 작은것을 더한것의 최댓값
N = int(input())
nums = list(map(int,input().split()))
output = max(nums[0],nums[-1])
for i in range(1,N-1):
    output = max(output, nums[i]+min(nums[i-1], nums[i+1]))
print(output)


#B
#N : 애플파이개수 K:연속으로 먹을개수
N,K = map(int,input().split())
deli = list(map(int,input().split()))
deli = deli+deli[:K]
output, summ = sum(deli[:K]),sum(deli[:K])
for i in range(N):
    summ = summ - deli[i]+deli[i+K]
    output = max(output,summ)
print(output)


#C
#큰수가 최대한 1을 많이가져가고 다음숫자에서 1채워주면된다.
# 100001 :33
# 011110 : 30

# 100000 : 32
# 011111 : 31


#D 시간초과
import sys
input = sys.stdin.readline

N,M = map(int,input().split())
E,S,MM = [],[],[]
output = 0
for i in range(N):
    string = input().rstrip()
    for j in range(M):
        if string[j] == 'E':
            E.append([i,j])
        elif string[j] == 'S':
            S.append([i,j])
        else:
            MM.append([i,j])

for x1,y1 in E:
    for x2,y2 in [[i,j] for i,j in S if i>=x1 and j>=y1]:
        output += len([1 for i,j in MM if i>=x2 and j>=y2])
        if output >= (10**9 + 7):
            output %= (10**9 + 7)

print(output)



#E
N,K = map(int,input().split())
A = list(map(lambda x: int(x)%K,input().split()))
if sum(A)%K != 0:
    print('blobsad')
    exit()

# start, length, summ = 0,0,0
# for i in range(N):
#     summ += A[i]
#     if summ >= K:
#         length = i-start+1
#         tmpA = A[start:i+1]
#         for j in range(length):
#             for k in range(length):


#F
def isPrime(x):
    import math
    if(x == 2):
        return True
    if(x%2 == 0 or x == 1):
        return False
    for i in range(3,math.floor(math.sqrt(x)) + 1,2):
        if(x%i == 0):
            return False
    return True
K,Q = map(int,input().split())
output = []
A = list(map(int,input().split()))
for a in A:
    if isPrime(K):
        if a % K == 0:
            output.append(1)
        else:
            output.append(K)
        continue
    
    



#a * 1,2,3,4,5... 이 K의 배수인가



#H
def find(x):
    for i in range(x+1,N):
        if A[i] > A[x] :
            return i
    return 0

        
N = int(input())
A = list(map(int,input().split()))
x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11 = 0,0,0,0,0,0,0,0,0,0,0
X = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]


# for i in range(N-10):
#     x1 = i
#     for x,y in zip([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10], [x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]):
#         y = find(x)
#         if not y:
#             break
#     else:
        




#처음부터 11개 설정하고 뒤에 10개가 더있다쳐보자.
# x1,x2,x3,x4~x11 뒤에 10개
# 11 + 10(뒤의개수)*9(당겨오는애들) + 9*9 +... + 1*9 : 뒤의개수+1 + (뒤의개수 + ... + 1)*9



