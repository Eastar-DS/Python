import sys
input = sys.stdin.readline
i = input()
while i != '':
    s,l,n,e=0,0,0,0
    for o in list(i)[:-1]:
        if o==32:e+=1
        elif 97<=o<=122:s+=1
        elif 65<=o<=90:l+=1
        else:
            n+=1
    print(f'{s} {l} {n} {e}')
    i = ''
    i = input()

#2 여긴 list안쓰는게 더빠르네 원소가 적어서 그럴듯. 사실상 1등!
import sys
input = sys.stdin.readline
i = input()
while i != '':
    s,l,n,e=0,0,0,0
    for o in i[:-1]:
        if o==' ':e+=1
        elif 'a'<=o<='z':s+=1
        elif '0'<=o<='9':n+=1
        else:
            l+=1
    print(f'{s} {l} {n} {e}')
    i = ''
    i = input()