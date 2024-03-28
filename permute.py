from collections import deque
from typing import List

class Solution:
    def _backtrack(self, nums: List[int], ans: List[List[int]], idx: int, occupied: List[int], perm: List[int]) -> None:
        if len(perm)==len(nums):
            # perm实时变化，复制一份存入ans
            ans.append(perm[:])
            return
        
        for i, num in enumerate(nums):
            if occupied[i] or (i>0 and nums[i]==nums[i-1] and not occupied[i-1]):
                continue
            perm.append(num)
            occupied[i] = True
            self._backtrack(nums, ans, idx+1, occupied, perm)
            occupied[i] = False
            perm.pop()


    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        # 排序保证重复数字相邻
        nums.sort()
        n = len(nums)
        occupied = [0] * n
        ans = []
        perm = []
        self._backtrack(nums, ans, 0, occupied, perm)
        return ans
    
def main():
    s = Solution()
    res1 = s.permuteUnique([1,1,2])
    print(res1)
    res2 = s.permuteUnique([1,2,3])
    print(res2)

if __name__ == "__main__":
    main()