#2750
print('\n'.join(sorted([input() for _ in range(int(input()))], key= lambda x : int(x))))

import sys
print(''.join(sorted([sys.stdin.readline() for _ in range(int(sys.stdin.readline()))], key= lambda x : int(x))))