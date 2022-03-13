import itertools
L,C = map(int,input().split())
alphas = list(input().split())
alphas.sort()
for com in itertools.combinations(alphas,L):
    a,b = 0,0
    for alpha in com:
        #'aeiou'보다 ['a','e','i','o','u']가 빠르네
        if alpha in ['a','e','i','o','u']:
            a+=1
        else:
            b+=1
    if a>0 and b>1:
        print(''.join(com))