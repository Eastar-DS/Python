def solution1(numbers, target):
    answer = 0
    nums = numbers.copy()
    length = len(nums)
    if(length==0 and target == 0):
        answer = 1
    if(length != 0):
        num = nums[-1]
        answer += solution1(nums[:-1], target-num)
        answer += solution1(nums[:-1], target+num)
        
    return answer



#....?
from itertools import product
def solution11(numbers, target):
    l = [(x, -x) for x in numbers]
    s = list(map(sum, product(*l)))
    return s.count(target)





def solution2(n, computers):
    answer = 0
    def network(i,j):
        if(computers[i][j] != 1):
            return
        else:
            computers[i][j] = -1
            for k in range(0,n):
                if(computers[j][k] == 1):
                    network(j,k)
            
            if(computers[i][i] == 1):
                computers[i][i] = -1
            if(computers[j][j] == 1):
                computers[j][j] = -1
            
    for i in range(0,n):
        for j in range(0,n):
            if(computers[i][j] == 1):
                network(i,j)
                answer += 1
    return answer







