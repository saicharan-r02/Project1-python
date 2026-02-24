def moveZeroes(nums):
        c=0
        for i in range(len(nums)):
            if nums[i]!=0:
                nums[c],nums[i]=nums[i],nums[c]
                c+=1

nums=[2,3,0,20,0,0,1,1,2,2,0]
moveZeroes(nums)              
print(nums)  

nums=[0,3,4,5,0,1,0,0,2,3,4,0]
moveZeroes(nums)
print(nums)

nums=[0,0,1,0,0,2,0]
moveZeroes(nums)
print(nums)