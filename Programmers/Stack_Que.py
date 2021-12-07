
def solution(progresses, speeds):
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












































