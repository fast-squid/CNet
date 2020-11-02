#include <math.h>
#include <iostream>

typedef  float D_type;


typedef struct weight_data_struct
{
    D_type* data;

    int batch = 0;
    int out_channel = 0;   

    int in_channel = 0;  
    int height = 0;
    int width = 0;
} ds; 

typedef struct network_layer
{
    int padding[4]={1,1,1,1};
    int stride[2]={1,1};
    int dilation[2]={1,1};

}layer;

void layerInit(layer* layer, int pad, int stride, int dil)
{
    layer->padding[0]=pad;
    layer->padding[1]=pad;
    layer->padding[2]=pad;
    layer->padding[3]=pad;

    layer->stride[0]=stride;
    layer->stride[1]=stride;

    layer->dilation[0]=dil;   
    layer->dilation[1]=dil;
}

void DataReset(ds* data)
{
    free(data->data);
    data->out_channel=0;
    data->in_channel=0;
    data->width=0;
    data->height=0;
    data->batch=0;
}

void Printer(ds* output_data)
{
    for( int n = 0; n<output_data->out_channel; n++)
    {   
        for( int c=0; c<output_data->in_channel; c++)
        {
            for(int h=0; h<output_data->height; h++)
            {
                for(int w=0; w<output_data->width; w++)
                {
                    int index = n*output_data->in_channel*output_data->height*output_data->width
                                +c*output_data->height*output_data->width+h*output_data->width+w;
                    std::cout<<output_data->data[index]<<" ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
    }

    return;
}

void Printer2(ds* output_data)
{
    for( int n = 0; n<output_data->batch; n++)
    {   
        for( int c=0; c<output_data->out_channel; c++)
        {
            for(int h=0; h<output_data->height; h++)
            {
                for(int w=0; w<output_data->width; w++)
                {
                    int index = n*output_data->in_channel*output_data->height*output_data->width
                                +c*output_data->height*output_data->width+h*output_data->width+w;
                    std::cout<<output_data->data[index]<<" ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
    }
    return;
}

void InitParamter(ds* data, int o_channel, int i_channel, int height, int width, bool mod=false)
{
    data->out_channel=o_channel;
    data->in_channel = i_channel;
    data->height = height;
    data->width = width;
    data->data = (D_type*)malloc(sizeof(D_type)*(o_channel*i_channel*height*width));

    D_type temp=0;

    for(int oc =0; oc<o_channel; oc++)
    {
        for(int ic=0; ic<i_channel; ic++)
        {
            for(int h=0; h<height; h++)
            {
                for(int w=0; w<width; w++)
                {
                    if( mod )
                    {
                        temp = 1;
                    }
                    data->data[oc*i_channel*height*width + ic*height*width + h*width + w] = temp++;
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


void DirectConvolution(const ds* pad_input_data, const ds* filter, ds* output, int padding, int stride, int dilation)
{
    int total_size = output->batch* output->out_channel* output->height * output->width;
    if( total_size == 0)
    {
        std::cout<<"ERROR"<<std::endl;
        exit(-1);
    }

    output->data = (D_type*)malloc(sizeof(D_type)*total_size);

    for(int b =0; b < output->batch; b++)
    {
        for(int oc=0; oc < output->out_channel; oc++)
        {
            for(int oh = 0; oh< output->height; oh++)
            {
                for(int ow =0; ow< output->width; ow++)
                {
                    int output_index = b*output->in_channel*output->height*output->width
                                        + oc*output->height*output->width
                                        + oh*output->width
                                        + ow;
                    
                    output->data[output_index] = 0;
                    
                    // Filter Reduction Phase
                    for(int ic =0; ic<filter->in_channel; ic++)
                    {
                        for( int fh =0; fh<filter->height; fh++)
                        {
                            for(int fw=0; fw<filter->width; fw++)
                            {
                                int filter_index = oc*filter->in_channel*filter->height*filter->width
                                                + ic*filter->height*filter->width
                                                + fh*filter->width
                                                + fw;

                                int pad_index = + ic*pad_input_data->height*pad_input_data->width  //oc*pad_input_data->height*pad_input_data->width
                                                + oh*pad_input_data->width
                                                + ow
                                                + fh*pad_input_data->width
                                                + fw;


                                output->data[output_index] += pad_input_data->data[ pad_index ] * filter->data[filter_index];
                            }
                        }
                    }
                }
            }
        }
    }

    return;
}

void Convolution(const ds* input, const ds* filter, ds* output, const layer* config, int mode = 0)
{
    int stride_w = config->stride[0];
    int stride_h = config->stride[1];
    int dilation_w = config->dilation[0];
    int dilation_h = config->dilation[1];

    int batch = input->out_channel;
    int in_channel = input->in_channel;
    int in_height = input->height;
    int in_width = input->width;

    int number_of_filter = filter->out_channel;
    int channel = filter->in_channel;
    int kernel_height = filter->height;
    int kernel_width = filter->width;

    int dilated_kernel_height = (kernel_height-1)*dilation_h+1;
    int dilated_kernel_width = (kernel_width-1)*dilation_w +1;

    int pad_top = config->padding[0];
    int pad_right = config->padding[1];
    int pad_bottom = config->padding[2];
    int pad_left = config->padding[3];

    ds pad_temp;
    PaddingInputImage(input, pad_top, &pad_temp);
    // Get Padded Image.
    //Printer(&pad_temp);
    output->batch=batch;
    output->out_channel=number_of_filter;
    output->height=floor( (float)(in_height-dilated_kernel_height+pad_top+pad_bottom)/stride_h + 1 );
    output->width=floor( (float)(in_width-dilated_kernel_width+pad_right+pad_left)/stride_w + 1);

    // output shape

    if(mode == 0)
    {
        // Do DirectConvolution
        DirectConvolution(&pad_temp, filter, output, 1,1,1);
        DataReset(&pad_temp);
        //Printer2(output);
    }
    else if(mode == 1)
    {
        // Do DepthWiseConvolution
        DataReset(&pad_temp);
    }
    return;
}

