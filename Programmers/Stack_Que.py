
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



































