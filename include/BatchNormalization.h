#include <cmath>
#include <iostream>
#include "DataStruct.h"

// E -> moving-Mean
// Var -> mobing-Variance

// y = (gamma/( root( Var[x] + eps) ))*x + beta - gamma*E[x]/ ( root( Var[x] + eps ) ) 
// channel wise -> mean and var of batch*height*width

void BatchNorm(ds* X, D_type* gamma, D_type* beta, D_type eps, D_type* moving_mean, D_type* moving_var, D_type* output_data)
{


}