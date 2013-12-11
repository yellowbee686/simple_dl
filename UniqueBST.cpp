/*
本题求在给定n的情况下用1-n n个节点能够组成多少种不同的BST
其实可以用拆解的方法结合动态规划的思想 比如当n=4时 BST的数量为分别以1 2 3 4为根时的BST数量
而其中每个又等于左右子树BST数量的乘积 比如root=3时 left为1 2 可以有两种组合方法 而right为4 只有1种BST 
因此当root=3时BST总数为2*1=2 同理root=4时BST总数为nums[3]*1
由于n是参数 数组不确定要申请多少 可以用一个vector存储当节点数为1-n时的BST数量 
*/
#include <cstdio>
#include <vector>
using namespace std;

class Solution {
public:
	int numTrees(int n) {
		vector<int> nums;
		nums.push_back(1);//0
		nums.push_back(1);//1
		int sum;
		for (int i=2;i<=n;++i)
		{
			sum=0;
			for (int j=1;j<=i;++j)
				sum+=nums[j-1]*nums[i-j];
			nums.push_back(sum);
		}
		return nums[n];
	}
};

int main()
{
	int n;
	while (scanf("%d",&n)!=EOF)
	{
		Solution s;
		printf("%d\n",s.numTrees(n));
	}	
	return 0;
}