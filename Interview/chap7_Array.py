class Solution(object):
    #1
    #my solution
    def twoSum1(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        length = len(nums)
        if(length < 2):
            return None
        for i in range(length - 1) :
            for j in range(i+1, length):
                if(nums[i] + nums[j] == target):
                    return [i,j]
        return None
    
    
    def twoSum2(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # in 사용. 
        # complement의 index를 표현할때 .index()사용과 (i+1을 더해주는 것 주의)
        for i, n in enumerate(nums):
            complement = target - n
            if complement in nums[i+1:]:
                return [i, nums[i+1:].index(complement) + (i+1)]
        
    
    def twoSum3(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        #hash table로 만들어진 dictionary를 이용
        dic = {}
        for i , num in enumerate(nums):
            dic[num] = i
            
        for i , num in enumerate(nums):
            if target - num in dic and i != dic[target - num]:
                return [i, dic[target - num]]
        '''
        조회구조 개선 : for문 하나에 묶어보자. 
        순서를 거꾸로 조회하는것과 같지만 코드가 더 간결해짐
        dic = {}
        for i , num in enumerate(nums):
            if target - num in dic and i != dic[target - num]:
                return [i, dic[target - num]]
            dic[num] = i
        '''
    
    
    #42
    def trap1(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        #내가 원하는 풀이 
        '''
        1. 제일큰수를 잡고 양옆으로 큰수를 찾아서 그사이의 모든 넓이에서 블록의수를뺌.
        2. 맨왼쪽부터 포인터를 이동하면서 첫 높이를 기준으로 더 큰높이를 만나게되면 
          부피를 계산해서 더해줌. -> 맨처음에 가장 큰높이면 어떻게? 어려워보임.
        '''
    
    
    def trap2(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        #투포인터
        if not height:
            return 0
        
        volume = 0
        left, right = 0, len(height) - 1
        left_max, right_max = height[left], height[right]
        
        while left < right:
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    