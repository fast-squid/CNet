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
 * Architecture :
 * network 
 *	|- layer
 *		|- sublayer
 *			|- operation(conv, bn, relu, dropout, linear...)
 */

typedef struct operation_
{
	int opcode = -1;
	int op_idx = -1;
	ds* filter;
	Param* param;
	ds (*op_fn)(ds* ,ds* , Param*) = NULL;
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

void InitOperation(operation* op, int opcode)
{
	op->opcode = opcode;
	if(op->opcode == CONV)
	{
		op->op_fn = Convolution;
	}
	else if(op->opcode == BN)
	{
		op->op_fn = BatchNormalization;
	}
	else if(op->opcode == RELU)
	{
		op->op_fn = Relu6;
	}
	else if(op->opcode == DROP)
	{
	}
	else if(op->opcode == LINEAR)
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
		printf("\t\t\t\toperation %d\n",i);
		//PrintMat(input_ptr);
		temp = sl->ops[i].op_fn(input_ptr, sl->ops[i].filter, sl->ops[i].param);
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
		printf("\t\tsublayer %d\n",i);
		temp = ForwardSublayer(&l->sublayers[i], input_ptr);
		input_ptr = &temp;
	}	
	if(input->out_channel == input_ptr->out_channel &&
			input->in_channel == input_ptr->in_channel &&
			input->height == input_ptr->height &&
			input->width == input_ptr->width && 
			l->sublayers[1].ops[0].param->strides == 1)
	{
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
