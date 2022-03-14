#dic
import sys
# input = sys.stdin.readline
A,B,C,D = {},{},{},{}
for _ in range(int(input())):
    a,b,c,d = map(int,input().split())
    if a in A:
        A[a] += 1
    else :
        A[a] = 1
    if b in B:
        B[b] += 1
    else :
        B[b] = 1
    if c in C:
        C[c] += 1
    else :
        C[c] = 1
    if d in D:
        D[d] += 1
    else :
        D[d] = 1

ab= {}
for a in A:
    for b in B:
        ab[a+b] = ab.get(a+b,0) + A[a]*B[b]
#get연산이 느리네.
output = 0
for c in C:
    for d in D:
        if -c-d in ab:
            output += C[c]*D[d]*ab[-c-d]
print(output)



#list 이용. 더빠르네;;?
import sys
# input = sys.stdin.readline
A,B,C,D = [],[],[],[]
for _ in range(int(input())):
    a,b,c,d = map(int,input().split())
    A.append(a);B.append(b);C.append(c);D.append(d);

ab= {}
for a in A:
    for b in B:
        ab[a+b] = ab.get(a+b,0) + 1

output = 0
for c in C:
    for d in D:
        if -c-d in ab:
            output += ab[-c-d]

print(output)



#리스트로 최대한 빠르게
import sys
# input = sys.stdin.readline
N = int(input())
A,B,C,D = [],[],[],[]
for _ in range(N):
    a,b,c,d = map(int,input().split())
    A.append(a);B.append(b);C.append(c);D.append(d);
#해주는게 더빠르다.
A.sort();B.sort();C.sort(reverse = True);D.sort(reverse = True);
#왜 append를 2**28이 넘는 수로 하나씩 더해줘야해?
ab = sorted([a+b for a in A for b in B])
ab.append(2**28+1)
cd = sorted([c+d for c in C for d in D],reverse = True)
cd.append(2**28+1)
i,j,k = 0,0,N*N
output = 0
while(i<k and j<k):
    summ = ab[i]+cd[j]
    if summ > 0:j+=1
    elif summ < 0:i+=1
    else:
        abx,cdy = ab[i],cd[j]
        x,y = i,j
        while(ab[i]==abx):i+=1
        while(cd[j]==cdy):j+=1
        output += (i-x)*(j-y)
print(output)



