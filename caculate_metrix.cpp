#include <stdio.h>
#include "first_svd.h"
#include "caculate_metrix.h"
#include "devide_metrix.h"
#include <print_metrix.h>
#include <multiMetrixCpu.h>
using namespace  std;
bool caculate_final_svd(vector<GPUSave*>&line)
{
	double* dev_Q;
	double* dev_q;
	double* dev_D;
	double* dev_W;
	bool right;
	FILE *ph;
	GPUSave * mnt;
	ph = fopen("F:/caculate_metrixs.txt","w");
	while(!line.empty())
	{
		mnt = line.back();
		dev_Q = (double *)malloc(sizeof(double )*mnt->size*(mnt->size+1));
		dev_q = (double*)malloc(sizeof(double)*(mnt->size+1));
		dev_D = (double*)malloc(sizeof(double)*mnt->size);
		dev_W = (double*)malloc(mnt->size*sizeof(double)*mnt->size);
		right=gpu_svd::first_svd(mnt->diagonal,mnt->offdiagonal,mnt->size,dev_Q,dev_q,dev_D,dev_W);
		if(!right)
		{
			printf("the svd in cpu is error,in caculate_metirx.cpp\n");
			return false;
		}
		right=tree_up_data(mnt->father,mnt,dev_Q,dev_q,dev_D,dev_W);
		if(!right)
		{
			printf("the svd in cpu is error,in caculate_metirx.cpp\n");
			return false;
		}
		delete mnt;
		line.pop_back();
	}
//	for (int i=0;i<line.size();i+=1)
//	{

//		/************************************************************************/
//		/*                               ต๗สิ                                       */
//		/************************************************************************/
//	//	print_mid_metrix(ph,dev_Q,dev_q,dev_D,dev_W,line[i]->size);

//		//get_result_test(ph,dev_Q,dev_D,dev_W,line[i]->size);
//		
//	//	right=gpu_svd::first_svd(line[i+1]->diagonal,line[i+1]->offdiagonal,line[i+1]->size,dev_Q,dev_q,dev_D,dev_W);
//	//	if (!right)
//	//	{
//	//		printf("the svd in cpu is error,in caculate_metirx.cpp\n");
//	//		return false;			
//	//	}
//	//	right=tree_up_data(line[i+1]->father,line[i+1],dev_Q,dev_q,dev_D,dev_W);	
//	//	if (!right)
//	//	{
//	//		printf("the svd in cpu is error,in caculate_metirx.cpp\n");
//	//		return false;			
//	//	}
//		delete line[i];
//	//	delete line[i+1];
//		/***********************************************************
//		/*ต๗สิ
//		/**************************************************************/
//		
//	//	print_mid_metrix(ph,dev_Q,dev_q,dev_D,dev_W,line[i+1]->size);
//		//get_result_test(ph,dev_Q,dev_D,dev_W,line[i+1]->size);


//	}
//	
	printf("\nright in caculate_metrix.cpp");
	getchar ();
	return true;
}

bool tree_up_data(GPUSave* father_point,GPUSave*son_point,double*dev_Q,double*dev_q,double*dev_D,double*dev_W)
{
	if(father_point==NULL||son_point==NULL)
	{
		printf("the tree_up_data is error, in caculate_metrix.cpp\n");
		return false;
	}
	if(son_point->leaf_left==true)
	{
		father_point->set_left_leaf(dev_Q,dev_q,dev_D,dev_W,son_point->size);
		father_point->left=NULL;
	}
	else
	{
		father_point->set_right_leaf(dev_Q,dev_q,dev_D,dev_W,son_point->size);
		father_point->right=NULL;
	}
	return true;
}