#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"
#include <string.h>


void PaddingInputImage(const ds* p_input, int pad ,ds* pad_temp)
{
    int out_channel = p_input->out_channel;
    int in_channel = p_input->in_channel;
    int pad_height = p_input->height + 2*pad;
    int pad_width = p_input->width + 2*pad;
	//InitMat(p_input, {out_channel, in_channel, pad_height, pad_width});

	int total_allocation_size = p_input->out_channel * p_input->in_channel * (p_input->height+2*pad) * (p_input->width+2*pad);

    pad_temp->data = (DTYPE*)malloc(sizeof(DTYPE)*total_allocation_size);
    pad_temp->out_channel=p_input->out_channel;
    pad_temp->in_channel=p_input->in_channel;
    pad_temp->height=p_input->height+2*pad;
    pad_temp->width=p_input->width+2*pad;

    for( int i0 =0; i0< out_channel; i0++)
    {   
        for(int i1 =0; i1< in_channel; i1++)
        {
            for(int i2=0; i2< pad_height; i2++)
            {
                for(int i3=0; i3< pad_width; i3++)
                {

                        int pad_index = i0*in_channel*pad_height*pad_width
                                        +i1*pad_height*pad_width
                                        +i2*pad_width
                                        +i3;
                        int input_index = i0*in_channel*p_input->height*p_input->width
                                        + i1*p_input->height*p_input->width
                                        + i2* p_input->width
                                        + i3 -(p_input->width*pad+pad);

                    if( ((pad <= i2)&&(i2 < pad_height-pad)) && ((pad <= i3)&&(i3 < pad_width-pad)) )
                    {
                        pad_temp->data[pad_index]= p_input->data[ input_index ];
                    }
                    else
                    {
                        pad_temp->data[pad_index]= 0.0f;
                    }
                }
            }
        }
    }
    return;
}

void SetOutputShape(ds* input, ds* filter, ds* output, Param* conv_p)
{
	output->out_channel = 1;
	output->in_channel = filter->out_channel;
	output->height = floor( (DTYPE)(input->height - filter->height +2*conv_p->padding)/conv_p->strides +1);
	output->width = floor( (DTYPE)(input->width - filter->width +2*conv_p->padding)/conv_p->strides +1 );
	//output->data = (DTYPE*)malloc(sizeof(DTYPE)*output->out_channel*output->in_channel*output->height*output->width); 
}

ds Convolution(ds* input, ds* filter, Param* conv_p )
{
	int groups = conv_p->groups;
	int padding = conv_p->padding;
	int strides = conv_p->strides;
	
	// init output
	ds output;
	SetOutputShape(input, filter, &output, conv_p);
	InitMat(&output, {output.out_channel, output.in_channel, output.height, output.width});
	// splitting by groups
	ds sliced_input = *input;
	ds sliced_output = output;
	ds sliced_filter = *filter;

	sliced_input.in_channel/=groups;
	sliced_filter.out_channel/=groups; // # of filters
	sliced_output.in_channel = sliced_filter.out_channel;

	int in_offset = sliced_input.in_channel
		*sliced_input.height
		*sliced_input.width;
	int out_offset = sliced_output.in_channel
		*sliced_output.height
		*sliced_output.width;	
	int filter_offset = sliced_filter.in_channel
		*sliced_filter.out_channel
		*sliced_filter.height
		*sliced_filter.width;
	
	for(int g = 0; g<groups; g++)
	{
		sliced_input.data = &input->data[g*in_offset];
		sliced_output.data = &output.data[g*out_offset];
		sliced_filter.data = &filter->data[g*filter_offset];
		
		ds pad_input;
		PaddingInputImage(&sliced_input, conv_p->padding, &pad_input);

		
		// ic,kh,kw ---> reduction index
		// output_d += Pad_input[ic][oh+kh][ow+hw]*Filter[ic][kh][kw]
		for(int oc=0; oc< sliced_output.in_channel; oc++ )
		{
			for(int oh=0; oh<sliced_output.height; oh++)
			{
				for( int ow=0; ow<sliced_output.width; ow++)
				{
					int out_index = oc*sliced_output.height*sliced_output.width
						+ oh*sliced_output.width
						+ ow;
					sliced_output.data[out_index] = 0;

					/// Reduction Phase
					for( int ic=0; ic< sliced_filter.in_channel; ic++)
					{
						for( int kh=0; kh<sliced_filter.height; kh++)
						{
							for( int kw=0; kw<sliced_filter.width; kw++)
							{
								int pad_index = ic*pad_input.height*pad_input.width
									+ oh*(conv_p->strides)*pad_input.height + kh*pad_input.height
									+ ow*(conv_p->strides) + kw;

								int kernel_index = oc*sliced_filter.in_channel*sliced_filter.height*sliced_filter.width
									+ ic*sliced_filter.height*sliced_filter.width
									+ kh*sliced_filter.width
									+ kw;
								sliced_output.data[out_index] +=  sliced_filter.data[kernel_index]* pad_input.data[pad_index] ;
							}
						}
					}
				}
			}
		}
		free( pad_input.data );

	}
	std::cout<<"\t\t\t\t\t\tConv done"<<std::endl;
    return output;
}


/*
Backup
void PaddingInputImage(const ds* p_input, int pad ,ds* pad_temp)
{
    int total_allocation_size = p_input->out_channel * p_input->in_channel * (p_input->height+2*pad) * (p_input->width+2*pad);

    pad_temp->data = (DTYPE*)malloc(sizeof(DTYPE)*total_allocation_size);
    pad_temp->out_channel=p_input->out_channel;
    pad_temp->in_channel=p_input->in_channel;
    pad_temp->height=p_input->height+2*pad;
    pad_temp->width=p_input->width+2*pad;

    int out_channel = p_input->out_channel;
    int in_channel = p_input->in_channel;
    int pad_height = p_input->height + 2*pad;
    int pad_width = p_input->width + 2*pad;

    for( int i0 =0; i0< out_channel; i0++)
    {   
        for(int i1 =0; i1< in_channel; i1++)
        {
            for(int i2=0; i2< pad_height; i2++)
            {
                for(int i3=0; i3< pad_width; i3++)
                {

                        int pad_index = i0*in_channel*pad_height*pad_width
                                        +i1*pad_height*pad_width
                                        +i2*pad_width
                                        +i3;
                        int input_index = i0*in_channel*p_input->height*p_input->width
                                        + i1*p_input->height*p_input->width
                                        + i2* p_input->width
                                        + i3 -(p_input->width*pad+pad);

                    if( ((pad <= i2)&&(i2 < pad_height-pad)) && ((pad <= i3)&&(i3 < pad_width-pad)) )
                    {
                        pad_temp->data[pad_index]= p_input->data[ input_index ];
                    }
                    else
                    {
                        pad_temp->data[pad_index]= 0.0f;
                    }
                }
            }
        }
    }
    return;
}

*/

void NaiveConvolution(ds* input, ds* filter, ds* output, Param* conv_p, int groups=1)
{

    output->out_channel = 1;
    output->in_channel = filter->out_channel;
    output->height = floor( (DTYPE)(input->height - filter->height +2*conv_p->padding)/ conv_p->strides +1);
    output->width = floor( (DTYPE)(input->width - filter->width +2*conv_p->padding)/ conv_p->strides +1 );	
    output->data = (DTYPE*)malloc(sizeof(DTYPE)*output->out_channel*output->in_channel*output->height*output->width);

    ds pad_input;
    PaddingInputImage(input, conv_p->padding, &pad_input);
    // ic,kh,kw ---> reduction index
    // output_d += Pad_input[ic][oh+kh][ow+hw]*Filter[ic][kh][kw]
	for(int g = 0; g<groups; g++)
	{
		// ic,kh,kw ---> reduction index
		// output_d += Pad_input[ic][oh+kh][ow+hw]*Filter[ic][kh][kw]
		for(int oc=0; oc< output->in_channel; oc++ )
		{
			for(int oh=0; oh<output->height; oh++)
			{
				for( int ow=0; ow<output->width; ow++)
				{
					int out_index = oc*output->height*output->width
						+ oh*output->width
						+ ow;
					output->data[out_index] = 0;

					/// Reduction Phase
					for( int ic=0; ic< filter->in_channel; ic++)
					{
						for( int kh=0; kh<filter->height; kh++)
						{
							for( int kw=0; kw<filter->width; kw++)
							{
								int pad_index = ic*pad_input.height*pad_input.width
									+ oh*(conv_p->strides)*pad_input.height + kh*pad_input.height
									+ ow*(conv_p->strides) + kw;

								int kernel_index = oc*filter->in_channel*filter->height*filter->width
									+ ic*filter->height*filter->width
									+ kh*filter->width
									+ kw;

								output->data[out_index] += pad_input.data[pad_index] * filter->data[kernel_index];
							}
						}
					}
				}
			}
		}
		free( pad_input.data );
	}
	std::cout<<"Conv done"<<std::endl;
    return;
}
