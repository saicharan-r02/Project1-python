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

nums=[2,3,9,2,4,5]
moveZeroes(nums)              
print(nums)  

nums=[0,30,40,5,0,1,0,0,0]
moveZeroes(nums)
print(nums)

nums=[0,0,33,0,92,0]
moveZeroes(nums)
print(nums)

nums=[2,3,99,0,9,0,9,0,1,0,2,0,3,0,4,0]
moveZeroes(nums)              
print(nums)  

nums=[0,3,4,5,0,0,2,3,6,7,0,0,8,9,0]
moveZeroes(nums)
print(nums)

nums=[0,0,0,0,1,2,3,0,0,0,4,5,6,0,0,0,7,8,9,0,0,0]
moveZeroes(nums)
print(nums)

