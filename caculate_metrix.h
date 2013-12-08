#include <stdio.h>
#include "devide_metrix.h"
bool caculate_final_svd(vector<GPUSave *>&line);
bool tree_up_data(GPUSave* father_point,GPUSave*son_point,double*dev_Q,double*dev_q,double*dev_D,double*dev_W);