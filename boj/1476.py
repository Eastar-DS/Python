#1
e,s,m = map(int,input().split())
for n in range(1,7981):
    E,S,M = n%15,n%28,n%19
    if E==0:
        E = 15
    if S==0:
        S = 28
    if M==0:
        M = 19
    if [E,S,M] == [e,s,m]:
        print(n)
        break
    
#2
e,s,m = map(int,input().split())
e,s,m = e%15,s%28,m%19
for n in range(1,7981):
    E,S,M = n%15,n%28,n%19
    if [E,S,M] == [e,s,m]:
        print(n)
        break

#3
e,s,m = map(int,input().split())
while(e!=s or s!=m):
    if e<=min(s,m):
        e+=15
    elif s<=min(e,m):
        s+=28
    else:
        m+=19
print(e)