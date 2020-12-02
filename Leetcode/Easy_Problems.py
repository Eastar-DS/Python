# class Solution:
#     def twoSum(self, nums: List[int], target: int) -> List[int]:


# class Solution:
    # def twoSum(self, nums: [int], target: int) -> list[int]:
    #     for i in range(len(nums)-1):
    #         start = i
    #         for j in range(len(nums)-(i+1)):
    #             Sum = nums[i] + nums[i+j+1]
    #             if((target - Sum) == 0):
    #                 Output = [i,i+j+1]
    #                 return Output
    



def twoSum(nums, target):
        for i in range(len(nums)-1):
            for j in range(len(nums)-(i+1)):
                Sum = nums[i] + nums[i+j+1]
                if((target - Sum) == 0):
                    Output = [i,i+j+1]
                    return Output