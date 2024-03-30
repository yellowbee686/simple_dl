#include <queue>
#include <vector>
#include <unordered_map>
using namespace std;

class SlidingWindow {
private:
    priority_queue<int> smallHeap; // 较小的那一半，大顶堆
    priority_queue<int, vector<int>, greater<int>> largeHeap; //较大的那一半，小顶堆
    // 记录真实大小
    int smallSize;
    int largeSize;
    unordered_map<int, int> delCount; //延迟删除，用于优化删除操作

    // 删除元素，直到top不在delCount中，采样template
    // 每次top移除，可能有新top顶上时都需要检查，确保此时的top是可用的，这样每次get时就不用再额外做操作
    template<typename T>
    void prune(T& heap) {
        while(!heap.empty()) {
            int toDel = heap.top();
            if (delCount[toDel] > 0) {
                delCount[toDel]--;
                heap.pop();
            } else {
                break;
            }
        }
    }

    void balance() {
        if (smallSize > largeSize + 1) {
            largeHeap.push(smallHeap.top());
            smallHeap.pop();
            --smallSize;
            ++largeSize;
            // small堆顶改变，顺便触发实际删除
            prune(smallHeap);
        } else if (smallSize < largeSize) {
            smallHeap.push(largeHeap.top());
            largeHeap.pop();
            ++smallSize;
            --largeSize;
            prune(largeHeap);
        }
    }
public:
    SlidingWindow(): smallSize(0), largeSize(0) {}

    double getMedian() {
        if (smallSize > largeSize) {
            return (double)smallHeap.top();
        } else {
            return (smallHeap.top() + largeHeap.top()) / 2.0;
        }
    }

    void insert(int num) {
        if (smallHeap.empty() || num <= smallHeap.top()) {
            smallHeap.push(num);
            smallSize++;
        } else {
            largeHeap.push(num);
            largeSize++;
        }
        balance();
    }

    void erase(int num) {
        delCount[num]++; //先加上，因为后面可能直接触发删除
        if (num <= smallHeap.top() || smallHeap.empty()) {
            smallSize--;
            // 如果直接在头部，直接触发删除
            if (num == smallHeap.top()) {
                prune(smallHeap);
            }
        } else {
            largeSize--;
            if (num == largeHeap.top()) {
                prune(largeHeap);
            }
        }
        // 已处理过，此时top都是有效的
        balance();
    }
};

class Solution {
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        SlidingWindow window;
        for (int i=0; i<k; i++) {
            window.insert(nums[i]);
        }
        const int n = nums.size();
        vector<double> ret(n - k + 1);
        for (int i = k; i<=n; i++) {
            ret[i-k] = window.getMedian();
            if (i < n) {
                window.insert(nums[i]);
                window.erase(nums[i-k]);
            }
        }
        return ret;
    }
};

int main() {
    vector<int> nums = {1,3,-1,-3,5,3,6,7};
    Solution* s = new Solution();
    vector<double> ret = s->medianSlidingWindow(nums, 3);
    return 0;
}