#include <stdio.h>

bool print_mid_metrix(FILE *ph,double *Q,double *q,double *D,double *W,int size);
bool print_original_metrix(FILE *ph,double* diagonal,double *offdiagonal,int size);
bool print_svd_metrix(FILE*ph,double *X,double *w,double *Y,int size);
bool print_block(FILE*ph,double * block,int rows,int cols);