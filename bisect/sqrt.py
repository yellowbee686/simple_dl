class Solution:
    def mySqrt(self, x: int) -> int:
        left = 0
        right = x
        while left < right:
            middle = (left+right) // 2
            res = middle * middle
            if res < x:
                left = middle + 1
            elif res > x:
                right = middle
            else:
                return middle
        return right - 1
    
s = Solution()
res_1 = s.mySqrt(8)
print(f'8 res:{res_1}')