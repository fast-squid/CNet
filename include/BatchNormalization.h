#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"
#include "Convolution.h"

#define EPS 0.00001
// E -> moving-Mean
// Var -> moving-Variance
// 
// y = (gamma/( root( Var[x] + eps) ))*x + beta - gamma*E[x]/ ( root( Var[x] + eps ) ) 
// channel wise -> mean and var of batch*height*width
// filter ( 1, 1, 1, width)
// filter.data -> moving_mean, moving_var, beta, gamma
ds BatchNormalization(ds* input, ds* filter, Param* conv_p)
{
	const DTYPE eps = EPS;
	ds output;
	InitMat(&output, {input->out_channel, input->in_channel, input->height, input->width});
	
	int size = input->out_channel*input->in_channel;
	DTYPE* moving_mean = &filter->data[0];
	DTYPE* moving_var = &filter->data[1*size];
	DTYPE* beta = &filter->data[2*size];
	DTYPE* gamma = &filter->data[3*size];

	DTYPE* factorA = (DTYPE*)malloc(sizeof(DTYPE)*size);
	DTYPE* var = (DTYPE*)malloc(sizeof(DTYPE)*size);	
	//printf("filter shape(%d %d %d %d)\n",filter->out_channel, filter->in_channel, filter->height, filter->width);		
	for(int i = 0; i < size; i++)
	{
		var[i] = sqrt(moving_var[i] + eps);
		factorA[i] = gamma[i]/var[i];
		//printf("%f | %f | %f | %f\n",moving_mean[i], moving_var[i],beta[i],gamma[i]);
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
                    output.data[data_index] = factorA[c_index]*(input->data[data_index] - moving_mean[c_index]) + beta[c_index];
                }
            }
        }
    }
	std::cout << "\t\t\t\t\t\tBatchNormalization Done" << std::endl; 
	return output;
}


