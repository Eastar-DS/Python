N = int(input())
nums1 = list(map(int,input().split()))
nums2 = nums1[::-1]
dp1,dp2=[1]*N,[1]*N
for i in range(N):
    for j in range(i):
        if nums1[j] < nums1[i] and dp1[j]+1>dp1[i]:
            dp1[i] = dp1[j]+1
    for k in range(i):
        if nums2[k] < nums2[i] and dp2[k]+1>dp2[i]:
            dp2[i] = dp2[k]+1
dp2 = dp2[::-1]
print(max([dp1[i]+dp2[i]-1 for i in range(N)]))