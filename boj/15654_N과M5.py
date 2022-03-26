import itertools
N,M = map(int,input().split())
nums = sorted(input().split(), key=lambda x : int(x))
for com in itertools.permutations(nums,M):
    print(' '.join(com))