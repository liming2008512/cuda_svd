#pragma  once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

#define SUBPROBLEMSIZE 10
/************************************************************************/
/* 这个size是指col，不是row          
/* leaf_left  true left ,false right
/************************************************************************/
bool device_diagonal();

#ifndef   GPUSave
class GPUSave 
{
public:
	GPUSave * left;
	GPUSave *right;
	GPUSave *father;
	double *diagonal;
	double *offdiagonal;
	double ak;
	double bk;
	
	int size;
	int deep;
	bool leaf_left;

	double *  left_Q;
	double *  left_q;
	double *  left_D;
	double *  left_W;
	int left_size;

	double * right_Q;
	double * right_q;
	double * right_D;
	double * right_W;
	int right_size;


	double * self_Q;
	double * self_q;
	double * self_D;
	double * self_W;

	GPUSave();
	bool set_left_leaf(double *temp_left_Q,double *temp_left_q,double *temp_left_D,double *temp_left_W,int left_size);
	bool set_right_leaf(double *temp_right_Q,double *temp_right_q,double *temp_right_D,double *temp_right_W,int right_size);

};

 /************************************************************************/
 /* devide_metrix（）是一个总的函数，把大对角传给他，他先把这个传个 devide_metrix_tree 来分解成一个树
 /* 然后这个的函数devide_metrix_tree（）把双对角分解，保存在最底层，上面的层记录层数，还要记录父节点
 /* 父节点的作用是计算出svd后向上传递
 /* 然后read_metrix_tree（）来遍历这棵树，把最底层的保存在final_line里，供cpu计算
 /* 把每层的节点都记录在each_deep里，供向上传了以后再进行计算	
 /* size 都是diagonal的大小，所以 Q=(size)*(size+1) q = size+1 D = size*size W = size*size
 /************************************************************************/
template < class T >
void ClearVector( vector< T >& vt );
bool devide_svd_mid(GPUSave *root,int size,double *diagonal, double *offdiagonal );
bool devide_metrix_tree(int size , double * diagonal, double * offdiagonal , int deep, GPUSave * root  );
bool devide_metrix(double * diagonal, double * offdiagonal , int size,vector<vector<GPUSave*> >&each_deep_metrix,vector<GPUSave*>&final_deep_metrix);
bool read_metrix_tree(vector<vector<GPUSave *>> &each_deep, vector<GPUSave *> &final_line, GPUSave * root );
bool save_leaf_metrix(GPUSave *root, int size, int deep , double * diagonal , double * offdiagonal );
bool main_svd(double *diagonal, double *offdiagonal,int size);
#endif