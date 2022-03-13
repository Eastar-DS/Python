import sys,collections
# input = sys.stdin.readline
for _ in range(int(input())):
    before,after = map(int,input().split())
    queue = collections.deque([[before,'']])
    visit = [0]*10000
    visit[before] = 1
    
    while(queue):
        n,o = queue.popleft()
        
        if n==after:
            print(o)
            break
        
        D = (2*n)%10000
        S = (n-1)%10000
        L = (n%1000)*10 + (n//1000)
        R = (n%10)*1000 + (n//10)
        
        for num,order in zip([D,S,L,R],['D','S','L','R']):
            if not visit[num]:
                visit[num] = 1
                queue.append([num,o+order])