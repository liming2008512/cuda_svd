#include "first_svd.h"
#include <print_metrix.h>
#include <stdlib.h>
#include <linalg.h>
/***************************************************************************************
 * 下面是该类的具体实现
 * 功能只是第一次svd分解
 ***************************************************************************************/
using namespace gpu_svd;
/**************************************************************************************************
 *用alglib库进行奇异值分解
 *该程序是在cpu上进行的
 *_m行数，_n列数, _uneed _vneed _speed 不用考虑，这存在问题，是可以优化的，考虑是否能直接svd双对角的
 *************************************************************************************************/
bool gpu_svd::first_svd(double *diagonal,double *offdiagonal,int size,double *Q,double *q,double *D,double *W)
{
	alglib::real_2d_array a;
	alglib::ae_int_t _m=size+1;
	alglib::ae_int_t _n = size;
	alglib::ae_int_t _uneed = 2;
	alglib::ae_int_t _vneed = 2;
	alglib::ae_int_t _speed = 2;
	alglib::real_2d_array _u;
	alglib::real_2d_array _vt;
	alglib::real_1d_array _w;
	double *_metrix;
	_metrix = (double *)malloc(sizeof(double)*size*(size+1));
	if (diagonal==NULL||offdiagonal==NULL)
	{
		printf("error in the fist_svd function");
		return 0;
	}
	for(int i=0;(unsigned int)i<size+1;i++)
	{
		for(int j=0;j<size;++j)
		{
			if(i==j)
			{
				*(_metrix+i*size+j)=diagonal[i];
			}
			else if(i==(j+1))
			{
				*(_metrix+i*size+j)=offdiagonal[j];
			}
			else
			{
				*(_metrix+i*size+j)=0;
			}
		}
	}
	a.setcontent(_m,_n,_metrix);

	alglib::rmatrixsvd(a,_m,_n,_uneed,_vneed,_speed,_w,_u,_vt);
	
	gpu_svd::get_1d_fromalg(_w.length(),D,&_w);                    /*get the D*/
	gpu_svd::get_2d_fromalg(0,_u.rows(),0,_u.cols()-1,Q,&_u);  /*get the Q*/
	gpu_svd::get_2d_fromalg(0,_u.rows(),_u.cols()-1,1,q,&_u);  /*get the q*/
	gpu_svd::get_2d_fromalg(0,_vt.rows(),0,_vt.cols(),W,&_vt);
	return true;
}
bool gpu_svd::get_1d_fromalg(int col,double *metrix,alglib::real_1d_array *a)
{
	if(a->length()<col)
	{
		return 0;
	}
	for(int i=0;i<col;++i)
	{
		metrix[i]=(*a)[i];
	}
	return 1;
}


bool gpu_svd::get_2d_fromalg( int start_row,int row_num,int start_col,int col_num,double *metrix,alglib::real_2d_array *a)
{
	if(a->cols()<start_col+col_num||a->rows()<start_row+row_num)
	{
		return 0;
	}
	for(int i=0;i<row_num;i++)
	{
		for(int j=0;j<col_num;++j)
		{
			*(metrix+i*col_num+j)=(*a)[i+start_row][start_col+j];
		}
	}
	return 1;
}



