
def solution1(progresses, speeds):
    answer = []
    
    if(not progresses):
        return answer
    
    times = []
    for index, progress in enumerate(progresses):
        ele = (100 - progress)//speeds[index]
        if(ele == ((100 - progress)/speeds[index])):
            times.append(ele)
        else:
            times.append(ele+1)
    length = len(times)
    output = 1
    time = times[0]
    for i in range(1, length):
        if(time >= times[i]) :
            output += 1
        else:
            answer.append(output)
            time = times[i]
            output = 1
    answer.append(output)
    return answer


def solution11(progresses, speeds):
    """
    `zip()`을 이용해서 기능의 작업률과 속도를 합쳐서 계산이 쉽도록 했다.
    `-((p-100)//s)` 이 부분은 필요한 작업일수를 구하는 계산식이다. 
    음수(-)로 몫을 구한 다음 다시 양수로 바꿔주었는데 `math.ceil()`한 것과 동일하다. 
    `Q[i][0]` 부분은 작업이 끝나기까지 필요한 일수이며, 
        `Q[i][1]` 부분은 `Q[i][0]`일째에 배포 가능한 기능 수라고 보면 된다. 
    (Q = [... , [days, functions]]) 뒷 작업은 앞 작업이 끝나기까지 필요한 날짜와 비교해서 
        작으면 이미 앞작업에서 구했던 Q의 원소에서 기능수 부분에 +1 해주고 
        크면 list Q에 [필요한 일수, 기능수 = 1]의 형태로 새로 추가한다. 
    원소 개수 만큼 반복이 끝나면 배포 가능한 기능 수 부분만 잘라서 답을 리턴하면 된다.
    """
    Q=[]
    for p, s in zip(progresses, speeds):
        if len(Q)==0 or Q[-1][0]<-((p-100)//s):
            Q.append([-((p-100)//s),1])
        else:
            Q[-1][1]+=1
    return [q[1] for q in Q]



def solution2(priorities, location):
    from collections import deque
    answer = 1
    length = len(priorities)
    que = deque(priorities)
    while(que):
        pr = que.popleft()
        temp = [prs for prs in que if prs > pr]
        if(temp):
            que.append(pr)
            if(location > 0):
                location -= 1
            else:
                location = length - 1
        
        else:
            if(location > 0):
                location -= 1
                answer += 1
                length -= 1
            else:
                return answer            
            
    return answer

#any사용과 (i,p)를 큐에넣어 사용하는생각
def solution21(priorities, location):
    queue =  [(i,p) for i,p in enumerate(priorities)]
    answer = 0
    while True:
        cur = queue.pop(0)
        if any(cur[1] < q[1] for q in queue):
            queue.append(cur)
        else:
            answer += 1
            if cur[0] == location:
                return answer


def solution3(bridge_length, weight, truck_weights):
    answer = 1
    from collections import deque
    tw = deque(truck_weights)
    onbridge = []
    
    t = tw.popleft()
    onbridge.append([t,0])
    w = weight - t
    while(onbridge):
        # print('before ',answer, onbridge)
        for truck in onbridge[::-1]:
            truck[1] += 1
            if(truck[1] == bridge_length):
                w += truck[0]
                onbridge.pop(0)
                
        if(tw and tw[0] <= w):
            t = tw.popleft()
            onbridge.append([t,0])
            w -= t
            
        answer += 1
        # print('after ',answer, onbridge)    
    return answer




def solution4(prices):
    answer = []
    prices.pop()
    length = len(prices)
    for index, price in enumerate(prices):
        time = 0
        for i in range(index+1, length):
            if(prices[i] >= price):
                time += 1            
            else:
                answer.append(time+1)
                break
            if(i == length - 1):
                answer.append(time+1)                
            
    answer += [1,0]
    return answer


def solution41(prices):
    answer = [0] * len(prices)
    for i in range(len(prices)):
        for j in range(i+1, len(prices)):
            if prices[i] <= prices[j]:
                answer[i] += 1
            else:
                answer[i] += 1
                break
    return answer


#문제의도에 좀더 맞고 같은 O(n^2)지만 반절임.
from collections import deque
def solution42(prices):
    answer = []
    prices = deque(prices)
    while prices:
        c = prices.popleft()

        count = 0
        for i in prices:
            if c > i:
                count += 1
                break
            count += 1

        answer.append(count)

    return answer


#O(N)??
def solution43(p):
    """
    stack은 '주식 가격이 처음으로 떨어지는 지점을 아직 못찾은 가격의 index 모음'입니다. 
    i for문을 돌며 'stack에 남은 것들이 i 번째에 처음으로 가격이 떨어지는가?'를 매번 검사합니다. 
    이때 queue를 쓰지 않고 stack을 써서 역으로 index를 검사하는 이유는 
    stack 내 뒤쪽 것이 p[i]보다 가격이 같거나 작다면, 
    그 앞의 것들은 i index에서 최초로 가격이 떨어질리 없기에 굳이 검사하지 않고 break로 시간을 줄일 수 있기 때문입니다.
    
    stack에 price 를 저장해서 하려고 하니까 골치아팠는데, index를 저장하면 되는거였네요!
    
    스택에 시간을 저장하면서 prices[stack[-1]이랑 prices[now]를 비교하는겁니다!
                         
    else break 덕분에 O(n) 이 되었군요
    """
    ans = [0] * len(p)
    stack = [0]
    for i in range(1, len(p)):
        if p[i] < p[stack[-1]]:
            for j in stack[::-1]:
                if p[i] < p[j]:
                    ans[j] = i-j
                    stack.remove(j)
                else:
                    break
        stack.append(i)
    for i in range(0, len(stack)-1):
        ans[stack[i]] = len(p) - stack[i] - 1
    return ans





















