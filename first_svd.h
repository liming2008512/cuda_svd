#include <linalg.h>
#include <stdlib.h>
#include <stdio.h>



//此处等待优化，因为我是先把对角矩阵转化成大矩阵，填充0来完成，然后再奇异值分解，这个是很不好的，有简化的余地
namespace gpu_svd{

        /***********************************
		 * 第一次奇异值分解，分解最小的矩阵，用cpu的alglib库
		 * 首先是要先填充矩阵，把双对角填充成一个完整的矩阵，
		 * 再调用库函数求解
		 * diagonal 和 offdiagonal 分别是size-1 大小的 1维数组，是双对角的元素
		 * u,w,vt 是返回的奇异值分解后的矩阵
		 * first_svd()用来分解矩阵，给双对角的值，还有大小，传回去Q，q,D,W 
		 * get_2d_fromalg()是把real_2d_array里的东西读出来
		 * get_1d_fromalg()是把real_1d_array里的东西读出来
		 * 在这里，Q,q,D,W都自动给他申请空间，不用提前申请空间
		 ***********************************/

		 bool first_svd(double *diagonal,double *offdiagonal,int size,double *Q,double *q,double *D,double  *W);
		 bool get_2d_fromalg( int start_row,int row_num,int start_col,int col_num,double *metrix,alglib::real_2d_array *a);
		 bool get_1d_fromalg(int col,double *metrix,alglib::real_1d_array *a);

}	
