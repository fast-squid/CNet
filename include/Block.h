#pragma once
#include <iostream>
#include "DataStruct.h"
#include "Convolution.h"
#include "BatchNormalization.h"
#include "Activation.h"

#define CONV 0
#define BN 1
#define RELU6 2
//void Relu6(ds* input_data, ds* filter = NULL, ds* output, conv_param(lc)* layer = NULL)
//void Convolution(ds* input, ds* filter, ds* output, conv_param(lc)* layer )
//void BatchNorm(ds* input, ds* filter = (1,1,1,width), ds* output, conv_param(lc)* =NULL)

typedef struct sublayer_
{
	int sublayer_type = -1;
	int sublayer_idx = -1;
	void(*sublayer_fn)(ds* ,ds* ,ds*) = NULL;
}sublayer;

typedef struct block_
{
	int sublayer_num = 0;	// # of sublayer
	sublayer* sublayers = NULL;
}block;


void InitSubLayer(sublayer* sl, int sl_type, int sl_idx, ds* input, ds* filter, ds* output, conv_param* conv_p)
{
	sl->sublayer_type = sl_type;
	sl->sublayer_idx = sl_idx;
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
		sl->sublayer_fn = Relu;
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

void PushSubLayer(block* blk, lc* layer. int layer_idx)
{
	blk->sublayers[layer_idx] = *layer;
}

