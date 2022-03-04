#1
import collections
N = int(input())
isvisit = [0]*(N+1)
dic = {}
for _ in range(int(input())):
    a,b = map(int,input().split())
    if a in dic:
        dic[a].append(b)
    else:
        dic[a] = [b]
    if b in dic:
        dic[b].append(a)
    else:
        dic[b] = [a]

output = 0
queue = collections.deque([1])
while(queue):
    n = queue.popleft()
    if isvisit[n]:
        continue
    isvisit[n] = 1
    if n in dic:
        for num in dic[n]:
            queue.append(num)
for i in range(2,N+1):
    if isvisit[i]:
        output+=1
print(output)

#2
import collections
N = int(input())
isvisit = [0]*(N+1)
dic = collections.defaultdict(list)
for _ in range(int(input())):
    a,b = map(int,input().split())
    dic[a].append(b)
    dic[b].append(a)

output = 0
queue = collections.deque([1])
while(queue):
    n = queue.popleft()
    if isvisit[n]:
        continue
    isvisit[n] = 1
    if n in dic:
        for num in dic[n]:
            queue.append(num)
for i in range(2,N+1):
    if isvisit[i]:
        output+=1
print(output)

#3
import sys,collections
input = sys.stdin.readline
N = int(input())
isvisit = [0]*(N+1)
dic = collections.defaultdict(list)
for _ in range(int(input())):
    a,b = map(int,input().split())
    dic[a].append(b)
    dic[b].append(a)

output = 0
def dfs(num):
    isvisit[num] = 1
    for n in dic[num]:
        if not isvisit[n] :
            dfs(n)
dfs(1)
for i in range(2,N+1):
    if isvisit[i]:
        output+=1
print(output)