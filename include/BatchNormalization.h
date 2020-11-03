#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"

// E -> moving-Mean
// Var -> moving-Variance
// 
// y = (gamma/( root( Var[x] + eps) ))*x + beta - gamma*E[x]/ ( root( Var[x] + eps ) ) 
// channel wise -> mean and var of batch*height*width

void BatchNorm(ds* X, D_type* gamma, D_type* beta, D_type eps, D_type* moving_mean, D_type* moving_var, ds* Y)
{
	Y->out_channel = X->out_channel;
	Y->in_channel = X->in_channel;
	Y->height = X->height;
	Y->width = X->width;
	Y->data = (D_type*)malloc(sizeof(X->data));

	int size = X->out_channel*X->in_channel;
	D_type* factorA = (D_type*)malloc(sizeof(D_type)*size);
	
	for(int i = 0; i < size; i++)
	{
		factorA[i] = gamma[i]/sqrt(moving_var[i] + eps);
	}

	for(int oc = 0; oc<X->out_channel; oc++)
    {
        for(int ic=0; ic<X->in_channel; ic++)
        {
            for(int h=0; h<X->height; h++)
            {
                for(int w=0; w<X->width; w++)
                {
                    int data_index= oc*X->in_channel*X->height*X->width 
						+ ic*X->height*X->width 
						+ h*X->width 
						+ w;
					int c_index = oc*X->in_channel
						+ic;
                    Y->data[data_index] = factorA[c_index]*(X->data[data_index] - moving_mean[c_index]) + beta[c_index];
                }
            }
        }
    }
}
