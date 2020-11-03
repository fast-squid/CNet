#pragma once
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

