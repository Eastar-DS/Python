def primes(N):
    #primes 만들기
    primes = [0,0] + [1]*(N-1)
    for i in range(2,int(N**0.5)+1):
        if primes[i] == 1:
            for j in range(2*i,N+1,i):
                primes[j] = 0
    #output 만들기
    return [i for i in range(N+1) if primes[i]]

N = int(input())
primes = primes(N)
length = len(primes)
output, summ, now = 0,0,0
for i in range(length):
    summ += primes[i]
    if summ < N:
        continue    
    while(summ>N):
        summ -= primes[now]
        now += 1
    if summ == N:
        output += 1
print(output)