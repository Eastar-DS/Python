import collections

before,ans = '','123456780'
for _ in range(3):
    before += input().replace(' ','')

queue = collections.deque([[before,0]])
visit = {before:1}
while(queue):
    now,c = queue.popleft()
    if now == ans:
        print(c)
        break
    index = now.index('0')
    if index==2 or index==5:
        dis = [-3,-1,3]
    elif index==3 or index==6:
        dis = [-3,1,3]
    else:
        dis = [-3,-1,1,3]
        
    for di in dis:
        newi = index+di
        if 0<=newi<=8 :
            new = ''
            for i in range(9):
                if i==index:
                    new+=now[newi]
                elif i==newi:
                    new+=now[index]
                else:
                    new+=now[i]
            if new not in visit:
                queue.append([new,c+1])
                visit[new]=1
else:
    print(-1)