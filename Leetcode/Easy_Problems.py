# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:


# class Solution:
#     def twoSum(self, nums: [int], target: int) -> list[int]:
#         for i in range(len(nums)-1):
#             start = i
#             for j in range(len(nums)-(i+1)):
#                 Sum = nums[i] + nums[i+j+1]
#                 if((target - Sum) == 0):
#                     Output = [i,i+j+1]
#                     return Output
    



# def twoSum(nums, target):
#         for i in range(len(nums)-1):
#             for j in range(len(nums)-(i+1)):
#                 Sum = nums[i] + nums[i+j+1]
#                 if((target - Sum) == 0):
#                     Output = [i,i+j+1]
#                     return Output
                
                
class Solution(object):
    def twoSum(nums, target):
        for i in range(len(nums)-1):
            for j in range(len(nums)-(i+1)):
                Sum = nums[i] + nums[i+j+1]
                if((target - Sum) == 0):
                    Output = [i,i+j+1]
                    return Output
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        
    def reverse(x):
        string = str(x)
        size = len(string)
        answer = ''
        if(x>2**31-1 or x<-2**31):
            return 0
        elif(x > 0):
            for i in range(size):
                answer = answer + string[size-(i+1)]
            if(int(answer)>(2**31)-1 or int(answer)<-(2**31)):
                return 0
        elif(x < 0):
            answer = '-'
            for i in range(size-1):
                answer = answer + string[size-(i+1)]
            if(int(answer)>(2**31)-1 or int(answer)<-(2**31)):
                return 0
        elif(x == 0):
            return 0;
        return int(answer)
        
        """
        :type x: int
        :rtype: int
        """
        
    def isPalindrome(x):
        """
        :type x: int
        :rtype: bool
        """
        if(x > 2**31 - 1 or x < -2**31):
            return 0
        string = str(x)
        size = int((len(string)+1)/2)
        for i in range(size):
            if(string[i] != string[len(string)-(i+1)]):
                return False
        return True    
        
        
        