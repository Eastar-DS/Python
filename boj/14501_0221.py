#1
import collections
N = int(input())
cost = [list(map(int,input().split())) for _ in range(N)]
queue = collections.deque()
for day in range(N):
    if day + cost[day][0] <= N:
        queue.append([day,cost[day][1]])

output = 0
while(queue):
    day,money = queue.popleft()
    output = max(money,output)
    for next_day in range(day+cost[day][0],N):
        if next_day + cost[next_day][0] <= N:
            queue.append([next_day, money+cost[next_day][1]])
print(output)


#2 dp로 나중에 다시풀자.
