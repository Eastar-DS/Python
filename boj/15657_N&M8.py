import itertools
N,M = map(int,input().split())
nums = sorted(input().split(), key=lambda x : int(x))
for com in itertools.combinations_with_replacement(nums, M):
    print(' '.join(com))