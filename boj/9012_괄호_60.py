for _ in range(int(input())):
    stack = []
    for p in input().rstrip():
        if p=='(':
            stack.append(p)
        else:
            if stack:
                stack.pop()
            else:
                print('NO')
                break
    else:
        if stack:
            print('NO')
        else:
            print('YES')
   

#2 stack을 숫자로해도되겠네
for _ in range(int(input())):
    check = 0
    for p in input().rstrip():
        if p=='(':
            check+=1
        else:
            if check:
                check-=1
            else:
                print('NO')
                break
    else:
        if check:
            print('NO')
        else:
            print('YES')