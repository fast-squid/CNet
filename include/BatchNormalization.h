#include <cmath>
#include <iostream>
#include "DataStruct.h"

// E -> moving-Mean
// Var -> mobing-Variance

// y = (gamma/( root( Var[x] + eps) ))*x + beta - gamma*E[x]/ ( root( Var[x] + eps ) ) 
// channel wise -> mean and var of batch*height*width

void BatchNorm(ds* X, ds* gamma, ds* beta, ds eps, ds* moving_mean, ds* moving_var, ds* output_data)
{


}