#include <test_alglib.h>
#include <print_metrix.h>
#include <time.h>
#include <multiMetrixCpu.h>
#define MALLOC(size) (double*)malloc(sizeof(double)*size)
bool test_alglib_print(void)
{
	double *Q;
	double *q;
	double *D;
	double *W;
	double *diagonal;
	double *offdiagonal;
	Q = MALLOC(6*5);
	q = MALLOC(6);
	D = MALLOC(5*5);
	W=MALLOC(5*5);
	diagonal =MALLOC(5);
	offdiagonal=MALLOC(5);
	for (int i=1;i<=5;++i)
	{
		diagonal[i-1]=i*5;
		offdiagonal[i-1]=i*3.0;
	}
	clock_t t;
	t=clock();
	gpu_svd::first_svd(diagonal,offdiagonal,5,Q,q,D,W);
	printf("\n the time %d\n",(clock()-t));
	FILE *ph;
	ph=fopen("D:/test_alglib.txt","w");
	print_mid_metrix(ph,Q,q,D,W,5);
	get_result_test(ph,Q,D,W,5);
	return true;
}