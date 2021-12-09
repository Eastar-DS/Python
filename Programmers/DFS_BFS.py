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














