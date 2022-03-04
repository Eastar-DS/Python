# N = int(input())
# if N==0:
#     print(0)
#     exit()
# output = ''
# while N != 1 :
#     output = str(N%2) + output
#     N = -(N//2)
    
# print('1'+output)

N = int(input())
if N==0:
    print(0)
else:
    output = ''
    while N != 1 :
        output += str(N%2)
        N = -(N//2)
    print((output+'1')[::-1])


