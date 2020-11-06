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
	ds* filter;
	conv_param* conv_p;
	ds (*sublayer_fn)(ds* ,ds* , conv_param*) = NULL;
}sublayer;

typedef struct block_
{
	int sublayer_num = 0;	// # of sublayer
	sublayer* sublayers = NULL;
}block;


void InitSubLayer(sublayer* sl, int sl_type, ds* filter, conv_param* conv_p)
{
	sl->sublayer_type = sl_type;
	sl->filter = filter;
	sl->conv_p = conv_p;
	if(sl->sublayer_type == CONV)
	{
		sl->sublayer_fn = Convolution_;
	}
	else if(sl->sublayer_type == BN)
	{
		sl->sublayer_fn = BatchNormalization_;
	}
	else if(sl->sublayer_type == RELU)
	{
		sl->sublayer_fn = Relu6_;
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

void PrintMat(const char* name, ds* mat)
{
	printf("%s : shape(%d,%d,%d,%d)\n",name, mat->out_channel, mat->in_channel, mat->height, mat->width);
}

ds ForwardBlock(block* blk, ds* input)
{
	ds output;
	ds* input_ptr = input;
	ds temp;
	for(int i = 0; i<blk->sublayer_num;i++)
	{
		PrintMat("input", input_ptr);
		temp = blk->sublayers[i].sublayer_fn(input_ptr, blk->sublayers[i].filter, blk->sublayers[i].conv_p);
		input_ptr = &temp;
		PrintMat("output", input_ptr);
	}
	output = *input_ptr;
	return output;
}
