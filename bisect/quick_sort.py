def partition(arr, low, high):
    """分区函数"""
    pivot = arr[high]
    i = low
    for j in range(low, high):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[high] = arr[high], arr[i]
    return i

def quicksort_inplace(arr, low, high):
    """原地快速排序"""
    if low < high:
        pi = partition(arr, low, high)
        quicksort_inplace(arr, low, pi - 1)
        quicksort_inplace(arr, pi + 1, high)

# 测试代码
if __name__ == "__main__":
    test_arr = [3, 6, 8, 10, 1, 2, 1]
    print("Original array:", test_arr)
    quicksort_inplace(test_arr, 0, len(test_arr) - 1)
    print("Sorted array:", test_arr)
