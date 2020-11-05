#pragma once
#include <iostream>
#include "DataStruct.h"
#include "Convolution.h"
#include "BatchNormalization.h"
#include "Activation.h"

#define CONV 0
#define BN 1
#define RELU 2
//void Relu6(ds* input_data, ds* filter = NULL, ds* output, conv_param(lc)* layer = NULL)
//void Convolution(ds* input, ds* filter, ds* output, conv_param(lc)* layer )
//void BatchNorm(ds* input, ds* filter = (1,1,1,width), ds* output, conv_param(lc)* =NULL)

typedef struct sublayer_
{
	int sublayer_type = -1;
	int sublayer_idx = -1;
	void(*sublayer_fn)(ds* ,ds* ,ds*, conv_param*) = NULL;
}sublayer;

typedef struct block_
{
	int sublayer_num = 0;	// # of sublayer
	sublayer* sublayers = NULL;
}block;


void InitSubLayer(sublayer* sl, int sl_type)
{
	sl->sublayer_type = sl_type;
	if(sl->sublayer_type == CONV)
	{
		sl->sublayer_fn = Convolution;
	}
	else if(sl->sublayer_type == BN)
	{
		sl->sublayer_fn = BatchNormalization;
	}
	else if(sl->sublayer_type == RELU)
	{
		sl->sublayer_fn = Relu6;
	}
	else
	{
		std::cout << "undefined sublayer error" << std::endl;
		exit(-1);
	}
}

void InitBlock(block *blk, int num)
{
	blk->sublayer_num = num;
	blk->sublayers = (sublayer*)malloc(sizeof(sublayer)*blk->sublayer_num);
}

void PushSubLayer(block* blk, sublayer* sl, int layer_idx)
{
	blk->sublayers[layer_idx] = *sl;
}

ds ForwardBlock(block* blk, ds* input, ds* filter, ds*output, conv_param* conv_p)
{
	ds* input_ptr = input; 
	ds* output_ptr = output;
	for(int i = 0; i<blk->sublayer_num;i++)
	{
		ds output;
		blk->sublayers[i].sublayer_fn(input_ptr, filter, &output, conv_p);
		input_ptr = &output;
	}
}
