import itertools
while 1:
    k,*S = input().split()
    k = int(k)
    if k==0:
        break
    for com in itertools.combinations(S,6):
        print(' '.join(com))
    print()