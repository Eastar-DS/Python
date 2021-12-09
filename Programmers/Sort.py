def solution1(array, commands):
    answer = []
    for [i,j,k]in commands:
        new_array = array[i-1:j]
        new_array.sort()
        answer.append(new_array[k-1])
    return answer


#대단한인간들
def solution11(array, commands):
    return list(map(lambda x:sorted(array[x[0]-1:x[1]])[x[2]-1], commands))








#x*3으로 비교하는걸 어떻게 생각한거지??? 캬...
def solution2(numbers):
    answer = ''
    strnum = [str(num) for num in numbers]
    strnum.sort(key = lambda x : x*3, reverse = True)
    #[0,0,0,0] -> '0000'
    if(strnum[0][0] == '0'):
        return '0'
    for num in strnum:
        answer += num
    
    return answer


def solution21(numbers):
    numbers = list(map(str, numbers))
    numbers.sort(key=lambda x: x*3, reverse=True)
    return str(int(''.join(numbers)))
    #반환 안될까봐 int안썻는데 굳이 마지막에 int로 바꿧다가 str하는건 별로같음.


from io import StringIO


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K:
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


def comparator(x, y):
    x = str(x)
    y = str(y)
    x_y = int(x + y)
    y_x = int(y + x)

    if x_y < y_x:
        return -1
    elif y_x < x_y:
        return 1
    else:
        return 0


def solution22(numbers):

    numbers = sorted(numbers, key=cmp_to_key(comparator), reverse=True)

    string_buffer = StringIO()
    for number in numbers:
        string_buffer.write(str(number))

    answer = int(string_buffer.getvalue())
    return str(answer)







def solution3(citations):
    for h in range(1000,-1,-1):
        papers = [paper for paper in citations if paper >= h]
        if(len(papers) >= h):
            return h


#answer = max(map(min, enumerate(citations, start=1))) 를 생각한게 대박이다. 
#속도는 말할것도없이 수십배빠름.
def solution31(citations):
    citations.sort(reverse=True)
    answer = max(map(min, enumerate(citations, start=1)))
    return answer

#위의 풀이를 한줄로 줄이기
def solution32(citations):    
    return max(map(min, enumerate(sorted(citations, reverse=True), start=1)))













