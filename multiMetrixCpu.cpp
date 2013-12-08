#include "multiMetrixCpu.h"
#include <print_metrix.h>
#include <stdlib.h>
double *trans(double *a,int ia,int ja) /*compute the trans matrics of a */
{
	int i,j;
	double *tra=(double *)malloc(sizeof(double)*ja*ia);
	for(i=0;i<ja;i++)for(j=0;j<ia;j++)
		tra[i*ia+j]=a[j*ja+i];
	return tra;
}

bool multi(double *a,double *b,double*result,int rowa,int cola,int colb) /*compute matrics a multiply by b*/
{
	for(int i=0;i<rowa;i++)
	{
		for(int j=0;j<colb;j++)
		{
			result[i*colb+j]=0;
			for(int k=0;k<cola;k++)
			{	
				result[i*colb+j]+=(a[i*cola+k]*b[k*colb+j]);
			}
		}
	}
	return true;
}

bool get_result_test(FILE *ph,double *dev_Q,double *D,double *W,int size)
{
	double *mid;
	double *result;
	double *dev_d;
	dev_d = (double *)malloc(sizeof(double)*size*size);
	for(int i=0;i<size;++i)
	{
		for(int j=0;j<size;++j)
		{
			if(i==j)
			{
				dev_d[j+i*size] = D[i];
			}
			else
			{
				dev_d[j+i*size]=0;
			}
		}
	}
	mid =(double*)malloc(sizeof(double)*size*(size+1));
	result = (double*)malloc(sizeof(double)*size*(size+1));
	multi(dev_Q,dev_d,mid,size+1,size,size);
	multi(mid,W,result,size+1,size,size);
	fprintf(ph,"\nthe bidiagonal :\n");
	print_block(ph,result,size+1,size);
	free(result);
	free(mid);
	free(dev_d);
	return true;
}