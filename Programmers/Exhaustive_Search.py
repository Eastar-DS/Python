def solution1(answers):
    from collections import deque
    par1 = deque([1,2,3,4,5])
    par2 = deque([2,1,2,3,2,4,2,5])
    par3 = deque([3,3,1,1,2,2,4,4,5,5])
    score1, score2, score3 = 0, 0, 0
    problems = deque(answers)
    
    while(problems):
        p = problems.popleft()
        a1, a2, a3 = par1.popleft(), par2.popleft(), par3.popleft()
        if(a1 == p):
            score1 += 1
        if(a2 == p):
            score2 += 1
        if(a3 == p):
            score3 += 1
            
        par1.append(a1)
        par2.append(a2)
        par3.append(a3)
    
    
    answer = [1]
    if(score2 > score1):
        answer = [2]
    if(score2 == score1):
        answer.append(2)
    if(score3 > max(score1,score2)):
        answer = [3]
    if(score3 == max(score1,score2)):
        answer.append(3)
    
    return answer


def solution11(answers):
    pattern1 = [1,2,3,4,5]
    pattern2 = [2,1,2,3,2,4,2,5]
    pattern3 = [3,3,1,1,2,2,4,4,5,5]
    score = [0, 0, 0]
    result = []

    for idx, answer in enumerate(answers):
        if answer == pattern1[idx%len(pattern1)]:
            score[0] += 1
        if answer == pattern2[idx%len(pattern2)]:
            score[1] += 1
        if answer == pattern3[idx%len(pattern3)]:
            score[2] += 1

    for idx, s in enumerate(score):
        if s == max(score):
            result.append(idx+1)

    return result



def solution2(numbers):    
    def isPrime(num):
        if(num == 0 or num == 1):
            return False
        for i in range(2,int(num**(1/2))+1):
            if(num % i == 0):
                return False
        return True
    
    from itertools import permutations
    pers = []
    for i in range(1, len(numbers)+1):
        pers += list(permutations(numbers, i))
    nums = [int("".join(tup)) for tup in pers]
    nums = set(nums)
    print(nums)
    answer = 0
    for num in nums:
        if(isPrime(num)):
            answer += 1
            
    return answer


# |합집합 &교집합 -차집합 ^대칭차집합(합-교)
from itertools import permutations
def solution21(numbers):
    a = set()
    for i in range(len(numbers)):
        a |= set(map(int, map("".join, permutations(list(numbers), i + 1))))
    a -= set(range(0, 2))
    for i in range(2, int(max(a) ** 0.5) + 1):
        a -= set(range(i * 2, max(a) + 1, i))
    return len(a)



#이런문제는 쉽지~
def solution3(brown, yellow):
    landw = []
    brown2 = int(brown/2)
    for i in range(1,int(brown2//2) + 1):
        landw.append([brown2-i,i])

    for [l,w] in landw:
        if((l-2)*(w) == yellow):
            return [l,w+2]


























