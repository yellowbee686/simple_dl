/* 12 0 1 0 2 1 0 1 3 2 1 2 1*/
#include <cstdio>

class Solution {
public:
	int trap(int A[], int n) {
		int sum=0, max=0, maxidx=0;
		for (int i=0;i<n;++i)
		{
			if(A[i]>=A[maxidx])
				maxidx=i;
		}
		for (int i=0;i<maxidx;++i)
		{
			if (A[i]>=max)		
				max=A[i];		
			else		
				sum+=max-A[i];
		}
		max=0;
		for (int i=n-1;i>maxidx;--i)
		{
			if (A[i]>=max)		
				max=A[i];		
			else		
				sum+=max-A[i];
		}	
		return sum;
	}
};

int main()
{
	int n;
	int a[20];
	while (scanf("%d",&n)!=EOF)
	{
		for (int i=0;i<n;++i)
			scanf("%d",&a[i]);
		Solution s;
		printf("%d\n",s.trap(a,n));
	}	
	return 0;
}