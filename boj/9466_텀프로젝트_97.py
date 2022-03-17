def dfs(i):
    if visit[i]:
        return
    visit[i] = 1
    if nums[i] in path:
        index = path.index(nums[i])
        for num in path[index:]:
            nums[num] = 0
        return
    else:
        path.append(nums[i])
        dfs(nums[i])
import sys
# input = sys.stdin.readline
sys.setrecursionlimit(100001)
for _ in range(int(input())):
    n = int(input())
    nums = [0] + list(map(int,input().split()))
    visit = [0]*(n+1)
    output = 0
    for i in range(1,n+1):
        if not visit[i]:
            path = [i]
            dfs(i)
    for n in nums:
        if n: output += 1
    print(output)

# path를 함수에주면 메모리초과가 뜬다. path가 사라지지않고 메모리에 쌓이는듯.
# 제출
def dfs(i):
    global output
    if visit[i]:
        return
    visit[i] = 1
    if visit[nums[i]] and nums[i] in path:
        index = path.index(nums[i])
        output -= len(path[index:])
        return
    else:
        path.append(nums[i])
        dfs(nums[i])
import sys
# input = sys.stdin.readline
sys.setrecursionlimit(111111)
for _ in range(int(input())):
    n = int(input())
    nums = [0] + list(map(int,input().split()))
    visit = [0]*(n+1)
    output = n
    for i in range(1,n+1):
        if not visit[i]:
            path = [i]
            dfs(i)
    print(output)
    
#빠른풀이 지린다...
import sys
# input = sys.stdin.readline
for _ in range(int(input())):
    n = int(input())
    nums = [0] + list(map(int,input().split()))
    visit = [0]*(n+1)
    output = 0
    for i in range(1,n+1):
        if not visit[i]:
            now=after=i
            #처음 싸이클이 만나는지점까지 돌린다.
            while(not visit[now]):
                visit[now] = 1
                now = nums[now]
            #처음시작부터 만나는 지점까지의 학생은 조가없으므로 더해준다. 만나는지점부터 끝까지는 싸이클.
            while(after != now):
                output += 1
                after = nums[after]
    print(output)