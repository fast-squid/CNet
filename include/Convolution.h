#include <math.h>
#include <iostream>

typedef  float D_type;

typedef struct weight_data_struct
{
    D_type* data;
    int out_channel = 0; // # of Filter
    int in_channel = 0;  
    int height = 0;
    int width = 0;
} ds;

typedef struct layer_config
{
    int padding=1;
    int strides=1;
    int groups=1;
} lc;

void layerInit(lc* layer, int pad, int stride, int groups=1)
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
                    temp = temp % 1024;
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

void Convolution(ds* input, ds* filter, ds* output, lc* layer )
{
    output->out_channel = 1;
    output->in_channel = filter->out_channel;
    output->height = floor( (D_type)(input->height - filter->height +2*layer->padding)/ layer->strides +1);
    output->width = floor( (D_type)(input->width - filter->width +2*layer->padding)/ layer->strides +1 );
    output->data = (D_type*)malloc(sizeof(D_type)*output->out_channel*output->in_channel*output->height*output->width);

    ds pad_input;
    PaddingInputImage(input, layer->padding, &pad_input);

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
                            int pad_index = ic*input->height*input->width
                                            + oh*input->height + kh*input->height
                                            + ow +layer->strides-1 + kw + layer->strides-1;

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
    return;
}

