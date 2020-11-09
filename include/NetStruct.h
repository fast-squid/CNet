#pragma once
#include <iostream>
#include "DataStruct.h"
#include "Convolution.h"
#include "BatchNormalization.h"
#include "Activation.h"
#include "Skipconnection.h"

const int CONV = 0;
const int BN = 1;
const int RELU = 2;
const int DROP = 3;
const int LINEAR = 4;
const int NETSIZE = 19;
/*
 * Network 
 *	|- layer
 *		|- sublayer
 *			|- operation(conv,bn,relu)
 */

typedef struct operation_
{
	int op_type = -1;
	int op_idx = -1;
	ds* filter;
	conv_param* conv_p;
	ds (*op_fn)(ds* ,ds* , conv_param*) = NULL;
}operation;

typedef struct sublayer_
{
	int sublayer_idx = -1;
	int size = -1;	// # of op
	operation* ops = NULL;
}sublayer;

typedef struct layer_
{
	int layer_idx = -1;
	int size = -1;
	sublayer* sublayers = NULL;
}layer;

typedef struct net_
{
	char name[20];
	int size = -1;
	layer* layers = NULL;
}net;

void InitOperation(operation* op, int op_type, ds* filter, conv_param* conv_p)
{
	op->op_type = op_type;
	op->filter = filter;
	op->conv_p = conv_p;
	if(op->op_type == CONV)
	{
		op->op_fn = Convolution_;
	}
	else if(op->op_type == BN)
	{
		op->op_fn = BatchNormalization_;
	}
	else if(op->op_type == RELU)
	{
		op->op_fn = Relu6_;
	}
	else if(op->op_type == DROP)
	{
	}
	else if(op->op_type == LINEAR)
	{
	}
	else
	{
		std::cout << "undefined op error" << std::endl;
		exit(-1);
	}
}

void InitSublayer(sublayer *sl, int num)
{
	sl->size = num;
	sl->ops = (operation*)malloc(sizeof(operation)*sl->size);
}

void InitLayer(layer *l, int num)
{
	l->size = num;
	l->sublayers = (sublayer*)malloc(sizeof(sublayer)*l->size);
}

void InitNetwork(net* n, char* name, int num)
{
	strcpy(n->name,name);
	n->size = num;
	n->layers =  (layer*)malloc(sizeof(layer)*n->size);
}

void PushOperation(sublayer* sl, operation* op, int op_idx)
{
	sl->ops[op_idx] = *op;
}

void PushSublayer(layer* l, sublayer* sl, int sublayer_idx)
{
	l->sublayers[sublayer_idx] = *sl;
}

void PushLayer(net* n, layer* l, int layer_idx)
{
	n->layers[layer_idx] = *l;
}

void PrintMat(ds* mat)
{
	int size = mat->out_channel*mat->in_channel*mat->height*mat->width;
	printf("shape(%d,%d,%d,%d), mat[%d] = %f, mat[%d] = %f\n", mat->out_channel, mat->in_channel, mat->height, mat->width,0,mat->data[0], size-1, mat->data[size-1]);
}


ds ForwardSublayer(sublayer* sl, ds* input)
{
	ds output;
	ds* input_ptr = input;
	ds temp;
	for(int i = 0; i<sl->size;i++)
	{
		printf("\t\toperation %d\n",i);
		//PrintMat(input_ptr);
		temp = sl->ops[i].op_fn(input_ptr, sl->ops[i].filter, sl->ops[i].conv_p);
		input_ptr = &temp;
		//PrintMat(input_ptr);
	}
	output = *input_ptr;
	return output;
}

ds ForwardLayer(layer* l, ds* input)
{
	ds output;
	ds* input_ptr = input;
	ds temp;
	for(int i=0;i<l->size;i++)
	{
		printf("\tsublayer %d\n",i);
		temp = ForwardSublayer(&l->sublayers[i], input_ptr);
		input_ptr = &temp;
	}	
	if(input->out_channel == input_ptr->out_channel &&
			input->in_channel == input_ptr->in_channel &&
			input->height == input_ptr->height &&
			input->width == input_ptr->width && 
			l->sublayers[1].ops[0].conv_p->strides == 1)
	{
		printf("skip\n");
		output = Skipconnection(input, input_ptr);
	}
	else
	{
		output = *input_ptr;
	}
	return output;
}

ds Inference(net* network, ds* input)
{
	ds output;
	ds* input_ptr = input;
	ds temp;
	for(int i=0;i<NETSIZE;i++)
	{	
		printf("layer %d\n",i);
		temp = ForwardLayer(&network->layers[i], input_ptr);
		input_ptr = &temp;
	}
	output = *input_ptr;
	return output;
}
