#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"
#include "Convolution.h"

#define EPS 0.000001
// E -> moving-Mean
// Var -> moving-Variance
// 
// y = (gamma/( root( Var[x] + eps) ))*x + beta - gamma*E[x]/ ( root( Var[x] + eps ) ) 
// channel wise -> mean and var of batch*height*width
// filter ( 1, 1, 1, width)
// filter.data -> moving_mean, moving_var, beta, gamma
void BatchNormalization(ds* input, ds* filter, ds* output, conv_param* conv_p)
{
	const D_type eps = EPS;
	output->out_channel = input->out_channel;
	output->in_channel = input->in_channel;
	output->height = input->height;
	output->width = input->width;
	output->data = (D_type*)malloc(sizeof(D_type)*GetTotalSize(output));

	int size = input->out_channel*input->in_channel;
	D_type* moving_mean = &filter->data[0];
	D_type* moving_var = &filter->data[1*size];
	D_type* beta = &filter->data[2*size];
	D_type* gamma = &filter->data[3*size];

	D_type* factorA = (D_type*)malloc(sizeof(D_type)*size);
	D_type* var = (D_type*)malloc(sizeof(D_type)*size);	
	
	for(int i = 0; i < size; i++)
	{
		var[i] = sqrt(moving_var[i] + eps);
		factorA[i] = gamma[i]/var[i];
	}

	for(int oc = 0; oc<input->out_channel; oc++)
    {
        for(int ic=0; ic<input->in_channel; ic++)
        {
            for(int h=0; h<input->height; h++)
            {
                for(int w=0; w<input->width; w++)
                {
                    int data_index= oc*input->in_channel*input->height*input->width 
						+ ic*input->height*input->width 
						+ h*input->width 
						+ w;
					int c_index = oc*input->in_channel
						+ic;
                    output->data[data_index] = factorA[c_index]*(input->data[data_index] - moving_mean[c_index]) + beta[c_index];
                }
            }
        }
    }
	std::cout << "BatchNormalization Done" << std::endl; 
}
/*
void BatchNorm(ds* X, D_type* gamma, D_type* beta, D_type eps, D_type* moving_mean, D_type* moving_var, ds* Y)
{
	Y->out_channel = X->out_channel;
	Y->in_channel = X->in_channel;
	Y->height = X->height;
	Y->width = X->width;
	Y->data = (D_type*)malloc(sizeof(D_type)*Y->out_channel*Y->in_channel*Y->height*Y->width);

	int size = X->out_channel*X->in_channel;
	D_type* factorA = (D_type*)malloc(sizeof(D_type)*size);
	D_type* var = (D_type*)malloc(sizeof(D_type)*size);	
	
	for(int i = 0; i < size; i++)
	{
		var[i] = sqrt(moving_var[i] + eps);
		factorA[i] = gamma[i]/var[i];
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
*/
