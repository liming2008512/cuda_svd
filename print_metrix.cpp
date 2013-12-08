#include <stdio.h>
#include	"print_metrix.h" 

bool print_mid_metrix(FILE *ph,double *Q,double *q,double *D,double *W,int size)
{
	fprintf(ph,"the Q shows:\n");
	print_block(ph,Q,size+1,size);
	fprintf(ph,"the q shows:\n");
	print_block(ph,q,size+1,1);
	fprintf(ph,"the D shows:\n");
	print_block(ph,D,1,size);
	fprintf(ph,"the W shows:\n");
	print_block(ph,W,size,size);
	return true;
}

bool print_block(FILE *ph,double * block, int rows,int cols)
{
	for(int i=0;i<rows;++i){
		for(int j=0;j<cols;++j){
			fprintf(ph,"%lf ",*(block+i*cols+j));
		}
		fprintf(ph,"\n");
	}
	return true;
}
bool print_original_metrix(FILE*ph,double* diagonal,double *offdiagonal,int size)
{
	fprintf(ph,"the diagonal shows :\n");
	print_block(ph,diagonal,size,1);
	fprintf(ph,"the offdiagonal shows:\n");
	print_block(ph,offdiagonal,size,1);
	return true;
}

bool print_svd_metrix(FILE *ph,double *X,double *w,double *Y,int size)
{
	fprintf(ph,"the X shows :\n");
	print_block(ph,X,size+1,size+1);
	fprintf(ph,"the w shows :\n");
	print_block(ph,w,1,size);
	fprintf(ph,"the Y shows:\n");
	print_block(ph,Y,size,size);
	return true;
}