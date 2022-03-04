# import sys
n = int(input())
p1,p2 = map(int,input().split())
dic = {}
for _ in range(int(input())):
    a,b = map(int,input().split())
    dic[b] = a

output = 0
par1,par2 = [],[]
def find1(n):
    global output
    if n in dic:
        if dic[n] == p2:
            output = 2
        par1.append(dic[n])
        find1(dic[n])
def find2(n):
    global output
    if n in dic:
        if dic[n] == p1:
            output = 1
        par2.append(dic[n])
        find2(dic[n])
find1(p1)
find2(p2)

if output == 0:
    #조상이 없는경우도 처리해주자
    if len(par1)==0 or len(par2)==0:
        print(-1)
        exit()
    if par1[-1] != par2[-1]:
        print(-1)
        exit()
    length = min(len(par1),len(par2))
    index = -1
    for i in range(-1,-length-1,-1):
        if par1[i] != par2[i]:
            index = i+1
            break
    
    print(len(par1)+len(par2)+2*(index+1))
    
elif output == 1:
    print(par2.index(p1)+1)
else:
    print(par1.index(p2)+1)


#2
n = int(input())
p1,p2 = map(int,input().split())
dic = {}
for _ in range(int(input())):
    a,b = map(int,input().split())
    dic[b] = a

par1,par2 = [],[]
def find(n,i):
    if n in dic:
        if i==1:
            par1.append(dic[n])
            find(dic[n],1)
        else:
            par2.append(dic[n])
            find(dic[n],0)
find(p1,1)
find(p2,0)

if p1 in par2:
    print(par2.index(p1)+1)
elif p2 in par1:
    print(par1.index(p2)+1)
else:
    l1,l2 = len(par1), len(par2)
    if l1==0 or l2==0 or (par1[-1] != par2[-1]):
        print(-1)
    else:
        length = min(l1,l2)
        index = -1
        for i in range(-1,-length-1,-1):
            if par1[i] != par2[i]:
                index = i+1
                break        
        print(l1+l2+2*(index+1))
