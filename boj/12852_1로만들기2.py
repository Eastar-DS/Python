#1 BFS
import collections
N = int(input())
visited = [0] * N + [1]
queue = collections.deque([[N]])
while(queue):
    nums = queue.popleft()
    if nums[-1] == 1:
        print(len(nums) - 1)
        print(' '.join(map(str,nums)))
        break
    num = nums[-1]
    if num%3 == 0 and not visited[num//3]:
        visited[num//3] = 1
        queue.append(nums + [num//3])
    if num%2 == 0 and not visited[num//2]:
        visited[num//2] = 1
        queue.append(nums + [num//2])
    if num-1 > 0 and not visited[num-1]:
        visited[num-1] = 1
        queue.append(nums + [num-1])