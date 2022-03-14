def dfs(index):
    if index == length:
        for line in graph:
            print(' '.join(line))
        exit()
    i,j = zeros[index]
    nums = ['1','2','3','4','5','6','7','8','9']
    row = graph[i]
    col = [graph[x][j] for x in range(9)]
    box = [graph[(i//3)*3+x][(j//3)*3+y] for x in range(3) for y in range(3)]
    #(i,j)에 넣을 수 있는 숫자만 nums에 남기기
    for num in set(row+col+box):
        if num in nums:
            nums.remove(num)
    #graph에 넣을 수 있는 숫자가 없으면 return
    if not nums:
        return
    #nums안의 num마다 dfs실행
    for num in nums:
        graph[i][j] = num
        dfs(index+1)
        graph[i][j] = '0'
    
graph = [input().split() for _ in range(9)]
zeros= []
for i in range(9):
    for j in range(9):
        if graph[i][j]=='0':
            zeros.append((i,j))
length = len(zeros)

dfs(0)