#한번에 못품
for _ in range(int(input())):
    n = int(input())
    l1 = list(map(int,input().split()))
    l2 = list(map(int,input().split()))
    if n==1:
        print(max(l1[0],l2[0]))
    else:
        dp1,dp2 = [l1[0],l2[0]+l1[1]], [l2[0],l1[0]+l2[1]]
        for i in range(2,n):
            dp1.append(max(dp2[i-2], dp2[i-1])+l1[i])
            dp2.append(max(dp1[i-2], dp1[i-1])+l2[i])
        print(max(dp1[-1], dp2[-1]))

#dp1[-1] : l1[-1]을 포함하는 최댓값
for _ in range(int(input())):
    n = int(input())
    l1 = list(map(int,input().split()))
    l2 = list(map(int,input().split()))
    if n==1:
        print(max(l1[0],l2[0]))
    else:
        dp1,dp2 = [l1[0],l2[0]+l1[1]], [l2[0],l1[0]+l2[1]]
        for i in range(2,n):
            dp1[0],dp1[1],dp2[0],dp2[1] = dp1[1],max(dp2[0], dp2[1])+l1[i], dp2[1], max(dp1[0], dp1[1])+l2[i]
        print(max(dp1[1], dp2[1]))
        