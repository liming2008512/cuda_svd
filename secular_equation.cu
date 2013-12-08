#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "secular_equation.cuh"
#include <devide_metrix.h>
#include <math_functions.h>
#include <helper_cuda.h>
#include <string.h>
#include <print_metrix.h>
#include <device_functions.h>
#define MAXSHARE 2000
#define MINACCURACY 1.0e-15
#define PRECISION_F 1.0e-13
#define ACCURACYMIN 1.0e-14
#define PRECISION_RELATED_X 1.0e-14
#define PRECISION_INTERVAL 1.0e-14
#define TRUE 1
#define BLOCK_SIZE 16
#define PRECISION_DIVISION_LIMITATION_ZERO 1.0e-14
#define PRECISION_COMPARE_LIMITATION_ZERO 1.0e-2
#define BLOCKMIN(size)  (((size+255)/256)>192?192:((size+255)/256))
#define THREADMIN(size) (size>256?256:size)




bool   produce_mid_metrix(GPUSave *temp)
{
	double *temp_z;
	double *temp_d;
	double *mid_z;
	double *mid_d;
	double *temp_w;
	double *result_Q;
	double *mid_Q;
	int size;
	double temp_r;
	double temp_c0;
	double temp_s0;
	double *mid_W;
	double *result_W;

	size = temp->left_size+temp->right_size+1;
	temp->size = size;


	temp_z = (double *)malloc(sizeof(double)*size);
	temp_d = (double*)malloc(sizeof(double)*(size+1));
	mid_z = (double *)malloc(sizeof(double )*size);
	mid_d=(double*)malloc(sizeof(double)*(size+1));
	temp_w=(double*)malloc(sizeof(double)*size);
	
	/*求解中间矩阵*/
	bool rights = true;
	rights=formM(temp->ak,temp->bk,temp->left_Q,temp->left_q,\
		temp->left_D,temp->right_Q,temp->right_q,temp->right_D,\
		temp_z,temp_d,temp->left_size,temp->right_size,temp_r);
	if(rights ==false)
	{
		printf("Error: in secular_equation 58");
		return false;
	}
	/*求c0,s0*/
	if (temp_r<0)
	{
		printf("Error: the error in the function of produce_mid_metrix of the secular_equation\n ");
		return false;
	}
	if(temp_r<MINACCURACY)
	{
		temp_r= MINACCURACY;
	}
	temp_c0 = temp->ak*temp->left_q[temp->left_size]/temp_r;
	temp_s0 = temp->bk*temp->right_q[0]/temp_r;

	/*给d进行排序*/
	sortM(size,temp_z,temp_d,mid_z,mid_d);
	/*求w*/
	mid_d[size]=mid_d[size-1]+geometric_z(mid_z,size);
	temp_d[size]=mid_d[size];
	mid_function(mid_d,mid_z,temp_w,size);
	/*free*/
	free(mid_z);
	free(mid_d);
	/*构造Q W*/
	mid_Q = (double*)malloc(sizeof(double)*(size+1)*size);
	produce_Q(temp,mid_Q,temp_c0,temp_s0);
	mid_W = (double*)malloc(sizeof(double)*size*size);
	produce_W(temp,mid_W);
	/*校正z，求u，v，同时让u*Q   v*W */
	result_Q=(double*)malloc(sizeof(double)*size*(size+1));
	result_W=(double*)malloc(sizeof(double)*size*size);
	mid_function_two(temp_d,temp_z,temp_w,mid_Q,mid_W,result_Q,result_W,size);

	/*free*/
	free(temp_z);
	free(temp_d);
	free(mid_W);
	free(mid_Q);
	return true;
}



double geometric_z(double *mid_z,int size)
{
	double all=0;
	for(int i=0;i<size;i++)
	{
		all+=pow(mid_z[i],2);
	}
	return sqrt(all);
}
		














/****************************************************************************************************************************************************
**************************************中间函数，主函数通过中间函数沟通gpu，这个包括申请内存啥的，为的是让*********************************
**************************************主函数个子小一点，如果有问题它上面有很大原因************************************************************
****************************************************************************************************************************************************/



bool mid_function_two(double *temp_d,double * temp_z,double *temp_w,double *Q,double *W,\
	double *result_Q,double *result_W,int size)
{
	double *dev_d;
	double *dev_z;
	double *result_z;
	double *dev_u;
	double *dev_v;
	double *dev_w;
	double *mid_Q;
	double *mid_w;
	double temp_c0;
	double temp_s0;
	double *dev_Q;
	double *dev_W;
	double *dev_result_Q;
	double *dev_result_W;
	mid_Q = (double *)malloc(sizeof(double)*size*(size+1));
	mid_w =(double  *)malloc(sizeof(double)*size*size);

	checkCudaErrors(cudaMalloc((void**)&dev_d,sizeof(double)*size));
	checkCudaErrors(cudaMalloc((void**)&dev_z,sizeof(double)*size));
	checkCudaErrors(cudaMalloc((void**)&result_z,sizeof(double)*size));
	checkCudaErrors(cudaMalloc((void**)&dev_w,sizeof(double)*size));
	
	checkCudaErrors(cudaMemcpy(dev_d,temp_d,sizeof(double)*size,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_z,temp_z,sizeof(double)*size,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_w,temp_w,sizeof(double)*size,cudaMemcpyHostToDevice));
	
	adjust_z<<<BLOCKMIN(size),THREADMIN(size)>>>(dev_d,dev_w,result_z,dev_z,size);
	checkCudaErrors(cudaFree(dev_z));

	checkCudaErrors(cudaMalloc((void**)&dev_u,sizeof(double)*size*size));
	checkCudaErrors(cudaMalloc((void**)&dev_v,sizeof(double)*size*size));

	get_u_v<<<BLOCKMIN(size),THREADMIN(size)>>>(dev_w,result_z,dev_d,dev_u,dev_v,size);
	double *zs;
	zs = (double *)malloc(sizeof(double)*size);
	checkCudaErrors(cudaMemcpy(zs,result_z,sizeof(double)*size,cudaMemcpyDeviceToHost));
	FILE *phs=fopen("D:/mmmmmm.txt","w");
	print_block(phs,zs,1,size);
//	checkCudaErrors(cudaMalloc((void**)&dev_Q,sizeof(double)*size*(size+1)));
	//checkCudaErrors(cudaMalloc((void**)&dev_W,sizeof(double)*size*size));
//	checkCudaErrors(cudaMalloc((void**)&dev_result_Q,sizeof(double)*size*(size+1)));
	//checkCudaErrors(cudaMalloc((void**)&dev_result_W,sizeof(double)*size*size));
	
	//checkCudaErrors(cudaMemcpy(dev_Q,mid_Q,sizeof(double)*size*(size+1),cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(dev_W,mid_W,sizeof(double)*size*size,cudaMemcpyHostToDevice));
	
//	dim3 dimBLOCK()
	//Muld_metrix<<<BLOCKMIN(size),THREADMIN(size)>>>(dev_Q,dev_)
	
	/*测试的时候使用，这里就不取出来了，直接在里面让Q和u相乘*/
	//checkCudaErrors(cudaMemcpy(u,dev_u,sizeof(double)*size*size,cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(v,dev_v,sizeof(double)*size*size,cudaMemcpyDeviceToHost));
	

	/***********************构造Q，w*******************************************************************/
	checkCudaErrors(cudaFree(dev_u));
	return true;




}
bool mid_function(double *mid_d,double *mid_z,double *w,int size)
{
	double * dev_d;
	double * dev_z;
	double * dev_w; 
	FILE *ph = fopen("F:/secular.txt","w");

	checkCudaErrors(cudaMalloc((void**)&dev_d,sizeof(double)*(size+1)));
	checkCudaErrors(cudaMalloc((void **)&dev_z,sizeof(double)*size));
	checkCudaErrors(cudaMalloc((void**)&dev_w,sizeof(double)*size));
	checkCudaErrors(cudaMemcpy(dev_d,mid_d,(size+1)*sizeof(double),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_z,mid_z,sizeof(double)*size,cudaMemcpyHostToDevice));
	//devide_gpu_thread(MAXNUM,threads,block);
	double *temps =(double*)malloc(size);
	double *temps_dev;
	checkCudaErrors(cudaMalloc((void**)&temps_dev,size*sizeof(double)));
	secular_equation_small<<<5,10,size*2*sizeof(double)>>>(dev_z,dev_d,dev_w,size,temps_dev);
	checkCudaErrors(cudaMemcpy(w,dev_w,size,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(temps,temps_dev,size,cudaMemcpyDeviceToHost));
	for(int i=0;i<size;++i)
	{
		fprintf(ph," is %lf\n ",temps[i]);

	}
	printf("\nright");
	cudaFree(temps_dev);
	cudaFree(dev_d);
	cudaFree(dev_w);
	cudaFree(dev_z);
	free(temps);
	return 0;

}






bool  formM(double alpha,double beta,double *Q1,double *q1,\
	double *D1,double *Q2,double *q2,double *D2,double *z,double *d,\
	int up_number,int down_number,double &gama)

{/*The following is formM function.Input alpha,beta,and the first 2 matrix of B1's svd,and the first 2 matrix of B2's svd,and n*/
	/*so we can get the middle matrix M (also z[n] and d[n], in which d[0]=0)*/  

	double last1,first2;
	/*The fowlloing is form the releted matrix */
	
	if(q1==NULL||Q1==NULL||D1==NULL||Q2==NULL||q2==NULL||D2==NULL||z==NULL||d==NULL)
	{
		printf("Error: in the secular_equation of formM");
		return false;
	}

	last1=q1[up_number];   first2=q2[0]; /*last1,first2*/

	gama=sqrt((alpha*last1)*(alpha*last1)+(beta*first2)*(beta*first2));  /*gama*/

	/*Form matrix M (z[i] and d[i])*/
	z[0]=gama;
	for(int i=0;i<up_number;i++)
	{
		z[i+1]=alpha*Q1[(up_number)*(up_number)+i];
	}
	for(int i=0;i<down_number;i++)
	{
		z[i+up_number+1]=beta*Q2[i];   /*z[i] is the first columns of matrix M*/
	}
	d[0]=0;
	for(int i=0;i<up_number;i++)
	{	
		d[i+1]=D1[i];
	}/* the No.(i-1) singler of B1*/
	for(int i=0;i<down_number;i++)
	{
		d[i+up_number+1]=D2[i];
	}/* the No.(i-k) singler of B2*/
	return true;
}

/*sort z[n] and d[n] of Matrix M, and d[n] adds another element for the secular equation*/
bool  sortM(int n,double *z,double *d,double *znew,double*dnew)  
{
	int t,i,j;
	double zvalue=0;  

	/*Sort the d[] and z[] */
	double min;
	double *d2=(double *)malloc(n*sizeof(double));        /*d2[n] is a copy of d[n]*/
	int *minloc=(int *)malloc(n*sizeof(double));   /*with this fuction we can get the right order so as to fit for the matrics M*/
	for(i=0;i<n;i++)
		d2[i]=d[i];
	minloc[0]=0;
	for(i=1;i<n;i++)
	{
		for(j=1;j<n;j++)
			if(d2[j]>0)
			{
				min=d2[j];
				minloc[i]=j;
				break;
			} /*set the initial values for min and minloc*/
		for(t=1;t<n;t++)
			if((min>d2[t])&&(d2[t]>0))
			{
				min=d2[t];
				minloc[i]=t;
			}
		d2[minloc[i]]=0;
	}    /*get the sorted location and save it to minloc[]*/

	for(i=0;i<n;i++)dnew[i]=d[minloc[i]];
	for(i=0;i<n;i++)znew[i]=z[minloc[i]];   /*dnew[] and znew[]*/
	for(i=0;i<n;i++)zvalue+=z[i]*z[i];
	dnew[n]=dnew[n-1]+sqrt(zvalue);  /*dnew[n] is dnew[n-1] added with zvalue,use for solve the eqution*/
	free(d2);
	free(minloc);
	return true;
}





__global__ void secular_equation_small(double *z,double *d,double *w,int size,double *temps)
{
	extern __shared__ double sh_temp[];
	double * temp_d = sh_temp;
	double * temp_z = (double*)&temp_d[size+1];
	int thread_num;
	int block_num;
	thread_num=threadIdx.x;
	block_num = blockIdx.x;
	int IsSuccess;
	double mid_w;
	for(int i=thread_num ; i<size ; i+=blockDim.x)
	{
		temp_d[i]=d[i]*d[i];
		temp_z[i]=z[i]*z[i];
	}
//	double sum_d=0,sum_z=0;
//	for(int i=thread_num ; i<size ; i+=blockDim.x)
//	{
//		sum_d+=d[i]*d[i];
//		sum_z+=z[i]*z[i];
//	}
//	if(thread_num==0)
//	{
//		sum_d+=d[size]*d[size]; //????为啥要算这个？
//	}
	if(thread_num==0)
	{
		temp_d[size]=d[size]*d[size];
	}
	__syncthreads();
	for(int i=thread_num+block_num*blockDim.x;i<size; i+= blockDim.x * gridDim.x)
	{
		
		mid_w=GPU_FindOneRoot_Bisection(temp_d,temp_z,size,d[i],d[i+1],IsSuccess,temps[i]);
		if(IsSuccess==true)
			w[i]=mid_w;
		else
			w[i]=10;
		
	}
}

/*给的都是平方  d2,z2,w2，节约计算*/




/*
__device__ double add_secular(double *temp_d,double *temp_z,int size,double w)
{
	double res = 0;
	//TINT32  exponent1  = 0;
	//TFLOAT64 mantissa1 = 0;
	//TINT32  exponent2  = 0;
	//TFLOAT64 mantissa2 = 0;
	int  exponentdown  = 0;
	double mantissadown = 0;
	int  exponentup  = 0;
	double mantissaup = 0;
	double temp = 0;
	double up = 0;
	double down = 0;
	for (int i = 0; i < size; i++)
	{
		// Zk^2/(Dk^2-SIGMA^2)
		up = temp_z[i];
		//printf("%d\n",up);

		down = temp_d[i]-w;
		//Division Precision Protection
		if(abs(down) < PRECISION_DIVISION_LIMITATION_ZERO)
		{
			mantissadown = frexp(down,&exponentdown);
			mantissaup = frexp(up,&exponentup);
			
			//*down refer to the *temp
			mantissadown = mantissaup/mantissadown;
			exponentdown = exponentup-exponentdown;

			temp = mantissadown * pow(2.0,exponentdown);
			res += temp;

			//IsUnderflow = TRUE;
			//return 0;
		}
		else
		{
			//printf("%d\n",down);
			temp = up/down;
			//printf("%d\n",temp);
			res += temp;
		}
	}
	res += 1;

	return res;
}
*/





__device__  double  add_secular(double *temp_d,double *temp_z,int size,double w)
{
	double __equal=0;
	double __down;
	double __square_z;
	if(size<0)
	{
		return false;
	}
	for(int i=0;i<size;++i)
	{
		__down = temp_d[i]-w;
		if(__down<MINACCURACY&&__down>=0)
		{
			__down=MINACCURACY;
		}
		else if(__down>-MINACCURACY&&__down<=0)
		{
			__down=-MINACCURACY;
		}
		__square_z=temp_z[i];
		__equal+= __square_z/__down;
	}
	return  __equal+1.0;
}



__device__ double  GPU_FindOneRoot_Bisection(double  *d, double  *z, int  len_d_z, double  Ulimit, double  Dlimit,int  &IsSuccess,double &tempfs)
{
	// it need absolute precision to computing the exact root when Zi is very small
	// but double can only support related precision, so no check here, when Zi is very small
	// just return the value that close to the limitation of the interval.



	double  tempu = Ulimit, tempd = Dlimit;   //change right
	double  tempx = (tempd + tempu) / 2;
	double  tempf = 0;

	// unrecursion method
	tempf = add_secular(d,z,len_d_z,tempx*tempx);
	while(abs(tempf)>PRECISION_F)
	{
		if(tempf < 0)
		{
			// find in [tempx,tempu]
			tempd = tempx;
			tempx = (tempd+tempu)/2;
			if(abs(tempu - tempx) < PRECISION_RELATED_X*tempu)
			{IsSuccess = TRUE;return (tempx+tempu)/2;}
			tempf = add_secular(d,z,len_d_z,tempx*tempx); //change here
		}
		else
		{
			// find in [tempd,tempx]
			tempu = tempx;
			tempx = (tempd+tempu)/2;
			if(abs(tempx-tempd) < PRECISION_RELATED_X*tempu)
			{IsSuccess = TRUE;return (tempd+tempx)/2;}
			tempf = add_secular(d,z,len_d_z,tempx*tempx);
		}
	}

	IsSuccess = TRUE;
	tempfs = tempf;
	return tempx;
}






/*/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/
/*下面是求解 v u 校正z的函数////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////*/

__global__ void adjust_z(double *d,double *w,double *z,double *dev_z,int size)
{
	double __add_left= 0;
	double __add_mid=0;
	double __add_right=0;
	double __up=0;
	double __down=0;
	int thr = threadIdx.x+blockDim.x*blockIdx.x;
	double __add ;
	for(int j=thr;j<size;j+=blockDim.x*gridDim.x)
	{
		__add_mid = 0;
		__add_left = w[size-1]*w[size-1]-d[j]*d[j];
		for(int i=0;i<=j-1&&i<=size-2;++i)
		{ 
			__up=w[i]*w[i]-d[j]*d[j];
			__down = d[i]*d[i]-d[j]*d[j];
			if(fabs(__down)<ACCURACYMIN)
			{
				__down=__down>=0?ACCURACYMIN:(0-ACCURACYMIN);
			}
			__add_mid+=__up/__down;
		}
		__add_right = 0;
		for (int i=j;i<=size-2;++i)
		{
			__up=w[i]*w[i]-d[j]*d[j];
			__down =d[i+1]*d[i+1]-d[j]*d[j];
			if(fabs(__down)<ACCURACYMIN)
			{
				__down=__down>=0?ACCURACYMIN:(0-ACCURACYMIN);
			}
			__add_right+=__up/__down;
		}
		__add=__add_left*__add_mid*__add_right;
		__add=__add>0?__add:(0-__add);
		z[j] = sqrt(__add);
		z[j] = dev_z[j]>0?z[j]:(0-z[j]);
	}
}
__global__ void get_u_v(double *w,double *z,double *d,double *u,double *v,int size)
{
	int thr = threadIdx.x+blockIdx.x*blockDim.x;
	double __add_u;
	double __add_v;
	double __down;
	for(int i=thr;i<size;i+=blockDim.x*gridDim.x)
	{
		__add_u = add_for_u(d,w,z,size,i);
		__add_v = add_for_v(d,w,z,size,i);
		for(int j=1;j<size;++j)
		{
			__down = d[j]*d[j]-w[i]*w[i];
			if(fabs(__down)<ACCURACYMIN)
			{
				__down = __down>=0?ACCURACYMIN:(0-ACCURACYMIN);
			}
			u[i*size+j]=z[j]/__down/__add_u;
			v[i*size+j]=z[j]*d[j]/__down/__add_v;
		}
		__down = d[0]*d[0]-w[i]*w[i];
		if(fabs(__down)<ACCURACYMIN)
		{
			__down = __down>=0?ACCURACYMIN:(0-ACCURACYMIN);
		}
		u[i*size]=z[0]/__down/__add_u;
		v[i*size]=-1/__add_v;
	}
}

__device__ double add_for_u(double *d,double *w,double *z,int size,int position)
{
	double __add=0;
	double __down;
	for(int i=0;i<size;++i)
	{
		__down = d[i]*d[i]-w[position]*w[position];
		__down = __down*__down;
		if(fabs(__down)<ACCURACYMIN)
		{
			__down = __down>0?ACCURACYMIN:(0-ACCURACYMIN);
		}
		__add+=(z[i]*z[i]/__down);
	}
	return sqrt(__add);
}

__device__  double add_for_v(double *d,double *w,double *z,int size,int position)
{
	double __add = 1.0;
	double __down;
	double __up;
	for(int i=1;i<size;++i)
	{
		__up = d[i]*z[i];
		__up = __up*__up;
		__down = d[i]*d[i]-w[position]*w[position];
		__down = __down*__down;
		if(fabs(__down)<ACCURACYMIN)
		{
			__down = __down>0?ACCURACYMIN:(0-ACCURACYMIN);
		}
		__add+=__up/__down;
	}
	return (sqrt(__add));
}























/*//////////////////////////////////////////////////////////////矩阵相乘////////////////////////////////////////////////////////////*/


__global__ void Muld_metrix(float* A, float* B, int wA, int wB, float* C)
{ 

	int bx = blockIdx.x; 
	int by = blockIdx.y;    
	int tx = threadIdx.x; 
	int ty = threadIdx.y;    
	int aBegin = wA * BLOCK_SIZE * by;    
	int aEnd = aBegin + wA - 1;    
	int aStep = BLOCK_SIZE;     
	int bBegin = BLOCK_SIZE * bx;    
	int bStep = BLOCK_SIZE * wB;    
	float Csub = 0;   
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) 
	{        
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];      
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];        
		As[ty][tx] = A[a + wA * ty + tx]; 
		Bs[ty][tx] = B[b + wB * ty + tx];        
		__syncthreads();         
		for (int k = 0; k < BLOCK_SIZE; ++k) 
			Csub += As[ty][k] * Bs[k][tx];       
		__syncthreads(); 
	}     
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx; 
	C[c + wB * ty + tx] = Csub; 
}


bool produce_Q(GPUSave * temp,double *result,double c0,double s0)
{
	if(temp==NULL||result==NULL)
	{
		printf("Error: in the produce_Q of the secular_equation \n");
		return false;
	}
	int col_num = temp->left_size+temp->right_size+1;
	int row_num = col_num+1;
	int first_num = temp->left_size;
	int last_num = temp->right_size;
	memset(result,0,sizeof(double)*col_num*row_num);
	for(int i=0;i<first_num+1;++i)
	{
		result[i*col_num] = temp->left_q[i]*c0;
	}
	for(int i=0;i<last_num+1;++i)
	{
		result[(i+first_num+1)*col_num]=temp->right_q[i]*s0;
	}
	for(int i=0;i<first_num+1;++i)
	{
		for(int j=0;j<first_num;++i)
		{
			result[j+1+i*col_num] = temp->left_Q[j+i*first_num];
		}
	}
	for(int i=0;i<last_num+1;i++)
	{
		for (int j=0;j<last_num;++j)
		{
			result[j+first_num+1+(i+first_num+1)*col_num] = temp->right_Q[j+last_num*i];
		}
	}
	return true;
}


bool produce_W(GPUSave*temp,double *result)
{
	int first_num = temp->left_size;
	int last_num = temp->right_size;
	int size = first_num+last_num+1;
	if(temp==NULL||result==NULL)
	{
		printf("Error: in the produce_W of secular_equation\n");
		return false;
	}
	memset(result,0,sizeof(double)*size*size);
	for(int i=0;i<first_num;++i)
	{
		for(int j=0;j<first_num;++j)
		{
			result[i*size+j+1] = temp->left_W[i*first_num+j];
		}
	}
	result[first_num*size]=1;
	for(int i=0;i<last_num;++i)
	{
		for (int j=0;j<last_num;++j)
		{
			result[(i+1+first_num)*size+j+1+first_num] = temp->right_W[i*last_num+j];
		}
	}
	return true;
}

