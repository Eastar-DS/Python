#풀이1 624ms
N = int(input())

dp = [0]*(N+1)
for n in range(2,N+1):
    dp[n] = dp[n-1] + 1
    if n%3 == 0:
        dp[n] = min(dp[n], dp[n//3] + 1)
    if n%2 == 0:
        dp[n] = min(dp[n], dp[n//2] + 1)

print(dp[N])


#풀이2 100ms
from collections import deque
N = int(input())
queue = deque([[N,0]])
visited = [0]*(N+1)
while(queue):
    num,c = queue.popleft()
    if num == 1:
        print(c)
        break
    if visited[num]:
        continue
    visited[num] = 1
    if(num%3 == 0):
        queue.append([num//3,c+1])
    if(num%2 == 0):
        queue.append([num//2,c+1])
    queue.append([num-1,c+1])


#풀이3  68ms
info = {1: 0, 2: 1}

def find(n):    
    if n in info:
        return info[n]
    
    m = 1 + min(find(n // 2) + n % 2, find(n // 3) + n % 3)
    info[n] = m
    
    return m

N = int(input())
print(find(N))













