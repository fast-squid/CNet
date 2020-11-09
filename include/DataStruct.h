#pragma once
typedef  float D_type;

typedef struct weight_data_struct
{
    D_type* data=NULL;
    int out_channel = 0; // # of Filter
    int in_channel = 0;  
    int height = 0;
    int width = 0;
} ds;

typedef struct layer_weight
{
    ds weight={NULL,1,1,1,1};
    ds mean={NULL,1,1,1,1};
    ds var={NULL,1,1,1,1};
    ds gamma ={NULL,1,1,1,1};
    ds beta ={NULL,1,1,1,1};

} lw;

typedef struct conv_param_
{
    int padding=1;
    int strides=1;
    int groups=1;
} conv_param;

inline int GetTotalSize(ds* mat)
{
	return mat->out_channel * mat->in_channel * mat->height * mat->width; 
}

inline int GetMatSize(ds* mat)
{
	return mat->height * mat->width; 
}
