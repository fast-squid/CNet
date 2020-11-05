#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"

void layerInit(lc* layer, int pad, int stride, int groups)
{
    layer->padding=pad;
    layer->strides=stride;
    layer->groups = groups;
    return;
}

void InitParameter(ds* data, int out_channel ,int in_channel, int height, int width)
{
    data->out_channel = out_channel;
    data->in_channel = in_channel;
    data->height = height;
    data->width = width;
    data->data = (D_type*)malloc(sizeof(D_type)*out_channel*in_channel*height*width);

    int temp =1;
    for(int oc = 0; oc<out_channel; oc++)
    {
        for(int ic=0; ic<in_channel; ic++)
        {
            for(int h=0; h<height; h++)
            {
                for(int w=0; w<width; w++)
                {
                    int data_index= oc*in_channel*height*width + ic*height*width + h*width + w;
                    data->data[data_index] = (D_type)temp;
                    temp = temp % 24;
                    temp ++;
                }
            }
        }
    }
    return;
}

void PaddingInputImage(const ds* p_input, int pad ,ds* pad_temp)
{
    int total_allocation_size = p_input->out_channel * p_input->in_channel * (p_input->height+2*pad) * (p_input->width+2*pad);

    pad_temp->data = (D_type*)malloc(sizeof(D_type)*total_allocation_size);
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

void Convolution(ds* input, ds* filter, ds* output, lc* layer, int groups=1)
{
	if(layer->groups == 1)
	{
		output->out_channel = 1;
		output->in_channel = filter->out_channel;
		output->height = floor( (D_type)(input->height - filter->height +2*layer->padding)/ layer->strides +1);
		output->width = floor( (D_type)(input->width - filter->width +2*layer->padding)/ layer->strides +1 );	
		output->data = (D_type*)malloc(sizeof(D_type)*output->out_channel*output->in_channel*output->height*output->width);
	}
    std::cout<<"Output_Shape = "<<output->out_channel<<","<<output->in_channel<<","<<output->height<<","<<output->width<<std::endl;
    ds pad_input;
    PaddingInputImage(input, layer->padding, &pad_input);

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
                                            + oh*(layer->strides)*pad_input.width + kh*pad_input.width
                                            + ow*(layer->strides) + kw;

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
	std::cout<<"Conv done"<<std::endl;
    return;
}

void SetOutputShape(ds* input, ds* filter, ds* output, lc* layer)
{
	output->out_channel = 1;
	output->in_channel = filter->out_channel;
	output->height = floor( (D_type)(input->height - filter->height +2*layer->padding)/layer->strides +1);
	output->width = floor( (D_type)(input->width - filter->width +2*layer->padding)/layer->strides +1 );
	output->data = (D_type*)malloc(sizeof(D_type)*output->out_channel*output->in_channel*output->height*output->width*layer->groups); // multiplied by groups
}

void GroupConvolution(ds* input, ds* filter, ds* output, lc* layer)
{
	int groups = layer->groups;
	int padding = layer->padding;
	int strides = layer->strides;
	// init output
	SetOutputShape(input, filter, output, layer);

	// splitting by groups
	ds sliced_input = *input;
	ds sliced_output = *output;
	ds sliced_filter = *filter;

	sliced_input.in_channel/=groups;
	sliced_filter.out_channel/=groups; // # of filters
	sliced_output.in_channel = sliced_filter.out_channel;
	
	printf("sliced_input shape (%d,%d,%d,%d)->(%d,%d,%d,%d)\n",input->out_channel, input->in_channel, input->height, input->width,
			sliced_input.out_channel, sliced_input.in_channel, sliced_input.height, sliced_input.width);
	printf("sliced_filter shape (%d,%d,%d,%d)->(%d,%d,%d,%d)\n",filter->out_channel, filter->in_channel, filter->height, filter->width,
			sliced_filter.out_channel, sliced_filter.in_channel, sliced_filter.height, sliced_filter.width);
	printf("sliced_output shape (%d,%d,%d,%d)->(%d,%d,%d,%d)\n",output->out_channel, output->in_channel, output->height, output->width,
			sliced_output.out_channel, sliced_output.in_channel, sliced_output.height, sliced_output.width);

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
		sliced_output.data = &output->data[g*out_offset];
		sliced_filter.data = &filter->data[g*filter_offset];
		Convolution(&sliced_input, &sliced_filter, &sliced_output, layer);
	}
}


