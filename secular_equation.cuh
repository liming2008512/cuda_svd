#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <device_functions.h>
#include "devide_metrix.h"
#include <stdio.h>
#include <math.h>


/*主函数*/

bool   produce_mid_metrix(GPUSave *temp);





/*这个函数是用来构造中间矩阵的*/
bool  formM(double alpha,double beta,double *Q1,double *q1,\
	double *D1,double *Q2,double *q2,double *D2,double *z,double *d,\
	int up_number,int down_number,double &gama);

bool  sortM(int n,double *z,double *d,double *znew,double*dnew);
/*sort z[n] and d[n] of Matrix M, and d[n] adds another element for the secular equation*/

/*这个函数是求解w的中间变量，目的是看看结果对不对*/

/*两个中间函数*/
bool mid_function(double *mid_d,double *mid_z,double *w,int size);

bool mid_function_two(double *temp_d,double * temp_z,double *temp_w,double *Q,double *W,\
	double *result_Q,double *result_W,int size);


double geometric_z(double *mid_z,int size);









/*这个函数是用来求和的*/
__device__  double  add_secular(double *temp_d,double *temp_z,int size,double w);


/*这个函数是用来二分法的*/
/*这个都是给的平方关系，有助于提高速度，w2,z2,d2*/

__device__ double  GPU_FindOneRoot_Bisection(double  *d, double  *z, int  len_d_z, double  Ulimit, double  Dlimit,int  &IsSuccess,double &temps);


/*这个函数是用来求久期方程的*/
__global__ void secular_equation_small(double *z,double *d,double *w,int size,double *temps);



/*/////////////////////////////////////////////////////////校正z,求解u,v//////////////////////////////////////////////////////////////*/
__device__  double add_for_v(double *d,double *w,double *z,int size,int position);
__device__ double add_for_u(double *d,double *w,double *z,int size,int position);
__global__ void get_u_v(double *w,double *z,double *d,double *u,double *v,int size);
__global__ void adjust_z(double *d,double *w,double *z,double *dev_z,int size);


/***********************************矩阵相乘***************************************************************/
__global__ void Muld_metrix(float* A, float* B, int wA, int wB, float* C);


/***********************************构造矩阵************************************************************/
bool produce_Q(GPUSave * temp,double *result,double c0,double s0);
bool produce_W(GPUSave*temp,double *result);


