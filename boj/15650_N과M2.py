import itertools
N,M = map(int,input().split())
for com in itertools.combinations(list(map(str,range(1,N+1))),M):
    print(' '.join(com))