#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include <stdio.h>
#include <string.h>
#include "first_svd.h"
#include <Windows.h>
#include "devide_metrix.h"
#include "test_alglib.h"
#define  SUBPROBLENUM 10

int user_printf(int *metrix_size,int *gpu_number );//user input function
int metrix_init(double *metrix_up,double *metrix_down, int metrix_size); //init the function about the metrix size and value   the size if rows'number
int metrix_devise(double  *metrix_up,double  *metrix_down,int metrix_size);  //devise the function   

int main(void)
{
	int metrix_size;
	int gpu_number;
	double *metrix_up_diagonal;
	double *metrix_down_diagonal;
	user_printf(&metrix_size,&gpu_number);
	
	metrix_up_diagonal = (double*)malloc(sizeof(double )*metrix_size);
	metrix_down_diagonal = (double *)malloc(sizeof(double)*metrix_size);

	metrix_init(metrix_up_diagonal,metrix_down_diagonal,metrix_size);
	metrix_devise(metrix_up_diagonal,metrix_down_diagonal,metrix_size);
	/*释放了对角的内存*/
	free(metrix_up_diagonal);
	free(metrix_down_diagonal);
	printf("\nplease input space to exit the window");
	system("pause");
	return 0;
	
}

int metrix_devise(double *metrix_up,double  *metrix_down,int metrix_size)
{
	if(metrix_up==NULL||metrix_down==NULL)
	{
		printf("the function of metrix_devise is error in the file of the main.cu \n");
		return 0;
	}
	if(main_svd(metrix_up,metrix_down,metrix_size)==true)
	{
		return 1 ;
	}
	else
	{
		return 0;
	}
}

int metrix_init(double *metrix_up,double * metrix_down,int metrix_size)
{
	int i;
	if(metrix_size<=0||metrix_up==NULL||metrix_down==NULL)
	{
		return 0;
	}
	for(i=1;i<=metrix_size;++i)
	{
		metrix_up[i-1]=i*103.0;
		metrix_down[i-1]=i*105.0;
	}
	return 1;
}


int user_printf(int *metrix_size,int *gpu_number )
{
	int select_user;
	int trues=1;
	printf("*******************************************************************************\n");
	printf("* the fanction can deal with big metrix\n");
	printf("* there are some choices that you can select:\n");
	printf("* 1:the size of the metrix:\n");
	printf("* 2.the GPU number that you want to use:\n");
	printf("* 3.run the function\n");
	printf("*******************************************************************************\n");
	while(trues){
		printf("please int put the number:");
		scanf("%d",&select_user);
		switch(select_user)
		{

		case 1:
			{
				printf("please input the size of metrix: ");
				scanf("%d",metrix_size);
				break;

			}
		case 2:
			{
				printf("please input the number that you want to use: ");
				scanf("%d",gpu_number);
				break;
			}
		case  3:
			{
				trues=0;
				break;
			}
		default:
			{
				printf("Error:please intput the number of 1~3");
			}
		}
	}
	return 1;
}
