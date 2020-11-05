#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"


void relu6(ds* input_data)
{

    for(int i0=0; i0< input_data->out_channel; i0++)
    {
        for(int i1=0; i1<input_data->in_channel; i1++)
        {
            for(int i2=0; i2<input_data->height; i2++)
            {
                for(int i3=0; i3<input_data->width; i3++)
                {
                    int i_index= i0*input_data->in_channel*input_data->height*input_data->width
                                    + i1 * input_data->height*input_data->width
                                    + i2 * input_data->width
                                    + i3;

                    if( input_data->data[i_index] <= 0.0f )
                    {
                        input_data->data[i_index]=0;
                    }
                    else if( input_data->data[i_index] >= 6.0f )
                    {
                        input_data->data[i_index] = 6;
                    }
                }
            }
        }
    }
}