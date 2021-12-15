def solution11(participant, completion):
    #효율성 실패
    # while(completion):
    #     name = participant.pop()
    #     if(name not in completion):
    #         return name
    #     else:
    #         completion.remove(name)
    # return participant.pop()

    #합격
    P = sorted(participant)
    C = sorted(completion)
    for index, string in enumerate(C):
        if(P[index] != string):
            return P[index]
    return P[-1]
    
#솔루션 Counter사용, 해쉬함수사용, zip활용
import collections
def solution12(participant, completion):
    answer = collections.Counter(participant) - collections.Counter(completion)
    return list(answer.keys())[0]
    
    
def solution13(participant, completion):
    answer = ''
    temp = 0
    dic = {}
    for part in participant:
        dic[hash(part)] = part
        temp += int(hash(part))
    for com in completion:
        temp -= hash(com)
    answer = dic[temp]

    return answer







def solution2(phone_book):
    answer = True
    nums = sorted(phone_book, reverse = True) 
    
    for i in range(len(nums)-1):
        length = min(len(nums[i]), len(nums[i+1]))
        if(nums[i][:length] == nums[i+1][:length]):            
            return False
    return answer


#솔루션 : startswith 사용, 해쉬맵, 정규식

def solution21(phoneBook):
    phoneBook = sorted(phoneBook)

    for p1, p2 in zip(phoneBook, phoneBook[1:]):
        if p2.startswith(p1):
            return False
    return True

def solution22(phone_book):
    answer = True
    hash_map = {}
    for phone_number in phone_book:
        hash_map[phone_number] = 1
    for phone_number in phone_book:
        temp = ""
        for number in phone_number:
            temp += number
            if temp in hash_map and temp != phone_number:
                answer = False
    return answer

def solution23(phoneBook):
    import re
    for b in phoneBook:
        p = re.compile("^"+b)
        for b2 in phoneBook:
            if b != b2 and p.match(b2):
                return False
    return True



def solution3(clothes):
    answer = 1
    dic = {}
    for cloth in clothes:
        if(cloth[1] in dic):
            dic[cloth[1]] += 1
        else:
            dic[cloth[1]] = 1
    for value in dic.values():
        answer *= value+1
    
    return answer-1

#솔루션 : 카운터

def solution31(clothes):
    from collections import Counter
    from functools import reduce
    cnt = Counter([kind for name, kind in clothes])
    answer = reduce(lambda x, y: x*(y+1), cnt.values(), 1) - 1
    return answer





def solution4(genres, plays):
    answer = []
    from collections import defaultdict
    
    dic,temp = defaultdict(list),defaultdict(int)
    for i,g,p in zip(range(len(plays)),genres,plays):
        temp[g] += p
        dic[g].append([p,i])
        
    g_order = dict(sorted(temp.items(), key = lambda x : x[1], reverse = True))
    

    for g in g_order:
        if(len(dic[g]) >= 2):
            temp = sorted(dic[g], key = lambda x : x[0], reverse = True)
            answer.append(temp[0][1])
            answer.append(temp[1][1])
        else:
            answer.append(dic[g][0][1])
    
    return answer

















