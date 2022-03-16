#1 sort
N,K = map(int,input().split())
print(sorted(map(int,input().split()))[K-1])

#2 heapq 더느리네...
import heapq
N,K = map(int,input().split())
heap = []
for n in list(map(int,input().split())):
    heapq.heappush(heap,n)
for _ in range(K-1):
    heapq.heappop(heap)
print(heapq.heappop(heap))

#3 heapify 
import heapq
N,K = map(int,input().split())
heap = list(map(int,input().split()))
heapq.heapify(heap)
for _ in range(K-1):
    heapq.heappop(heap)
print(heapq.heappop(heap))