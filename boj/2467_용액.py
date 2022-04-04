#1 Two Pointer
l,r = 0,int(input())-1
nums = list(map(int,input().split()))
ansl,ansr,value = nums[l],nums[r],abs(nums[l]+nums[r])
while (l<r and value != 0):
    tmp = nums[l]+nums[r]
    if abs(tmp) < value:
        ansl,ansr,value = nums[l],nums[r],abs(tmp)
    if tmp < 0:
        l += 1
    elif tmp > 0:
        r -= 1
    else:
        break
print(f"{ansl} {ansr}")