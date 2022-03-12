def primes(n):
    output = [0,0] + [1]*(n-1)
    for i in range(2,n+1):
        if output[i]==0:
            continue
        for j in range(2*i,n+1,i):
            output[j]=0
    return output

import sys, collections
# input = sys.stdin.readline
primes = primes(10000)
nums = ['1','2','3','4','5','6','7','8','9']
for _ in range(int(input())):
    before,after = input().split()
    queue = collections.deque([[before,0]])
    visited = [0]*10000
    visited[int(before)] = 1
    while(queue):
        now,c = queue.popleft()
        if now==after:
            print(c)
            break
        #천의자리 바꾸기
        nums.remove(now[0])
        for num in nums:
            new = num+now[1:]
            if primes[int(new)] and not visited[int(new)]:
                queue.append([new,c+1])
                visited[int(new)] = 1
        nums.append(now[0])
        #100,10,1
        nums.append('0')
        nums.remove(now[1])
        for num in nums:
            new = now[0]+num+now[2:]
            if primes[int(new)] and not visited[int(new)]:
                queue.append([new,c+1])
                visited[int(new)] = 1
        nums.append(now[1])
        #
        nums.remove(now[2])
        for num in nums:
            new = now[:2]+num+now[3]
            if primes[int(new)] and not visited[int(new)]:
                queue.append([new,c+1])
                visited[int(new)] = 1
        nums.append(now[2])
        nums.remove(now[3])
        for num in nums:
            new = now[:3]+num
            if primes[int(new)] and not visited[int(new)]:
                queue.append([new,c+1])
                visited[int(new)] = 1
        nums.append(now[3])
        nums.remove('0')