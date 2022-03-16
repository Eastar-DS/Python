#1 counter
import sys, collections
# input = sys.stdin.readline
nums = collections.Counter([int(input()) for _ in range(int(input()))])
print(sorted(nums.most_common(), key=lambda x: (-x[1],x[0]))[0][0])

#2 dic
import sys
# input = sys.stdin.readline
dic = {}
for _ in range(int(input())):
    n = int(input())
    if n in dic:
        dic[n] += 1
    else:
        dic[n] = 1
print(sorted(dic.items(), key=lambda x: (-x[1],x[0]))[0][0])


#3 젤빠름
import sys
# input = sys.stdin.readline
dic = {}
for _ in range(int(input())):
    n = int(input())
    if n in dic:
        dic[n] += 1
    else:
        dic[n] = 1
output,tmp = 0,0
for k,v in dic.items():
    if v==tmp and k<output:
        output = k
    elif v>tmp:
        output,tmp = k,v
print(output)