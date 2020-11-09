#pragma once
#include <iostream>
#include "DataStruct.h"
#include "Convolution.h"
#include "BatchNormalization.h"
#include "Activation.h"
#include "NetStruct.h"

const int layer_sizes[] = {
	1,3,4,4,4,
	4,4,4,4,4,
	4,4,4,4,4,
	4,4,4,1,1
};

const int sublayer_sizes[] = {
	3,				
	3,1,1,			
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,3,1,1,		
	3,				
	2				
};

const int opcodes[] = {
	CONV,BN,RELU,							
	CONV,BN,RELU, CONV, BN,					
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	
	CONV,BN,RELU,							
	DROP,LINEAR								
};

/*
 * shapes[layer] = {n,c,h,w}
 * layer 0, 18 : ConvBNReLU
 * layer 1 : ConvBNReLU, Conv, BN
 * layer 2~ 17 : ConvBNReLU, ConvBNReLU, Conv, BN
 */
const int shapes_[][4] = {
	{32,3,3,3},{1,1,1,128},{0,0,0,0},
	{32,1,3,3},{1,1,1,128},{0,0,0,0},{16,32,1,1},{1,1,1,64},
	{96,16,1,1},{1,1,1,384},{0,0,0,0},{96,1,3,3},{1,1,1,384},{0,0,0,0},{24,96,1,1},{1,1,1,96},
	{144,24,1,1},{1,1,1,576},{0,0,0,0},{144,1,3,3},{1,1,1,576},{0,0,0,0},{24,144,1,1},{1,1,1,96},
	{144,24,1,1},{1,1,1,576},{0,0,0,0},{144,1,3,3},{1,1,1,576},{0,0,0,0},{32,144,1,1},{1,1,1,128},
	{192,32,1,1},{1,1,1,768},{0,0,0,0},{192,1,3,3},{1,1,1,768},{0,0,0,0},{32,192,1,1},{1,1,1,128},
	{192,32,1,1},{1,1,1,768},{0,0,0,0},{192,1,3,3},{1,1,1,768},{0,0,0,0},{32,192,1,1},{1,1,1,128},
	{192,32,1,1},{1,1,1,768},{0,0,0,0},{192,1,3,3},{1,1,1,768},{0,0,0,0},{64,192,1,1},{1,1,1,256},
	{384,64,1,1},{1,1,1,1536},{0,0,0,0},{384,1,3,3},{1,1,1,1536},{0,0,0,0},{64,384,1,1},{1,1,1,256},
	{384,64,1,1},{1,1,1,1536},{0,0,0,0},{384,1,3,3},{1,1,1,1536},{0,0,0,0},{64,384,1,1},{1,1,1,256},
	{384,64,1,1},{1,1,1,1536},{0,0,0,0},{384,1,3,3},{1,1,1,1536},{0,0,0,0},{64,384,1,1},{1,1,1,256},
	{384,64,1,1},{1,1,1,1536},{0,0,0,0},{384,1,3,3},{1,1,1,1536},{0,0,0,0},{96,384,1,1},{1,1,1,384},
	{576,96,1,1},{1,1,1,2304},{0,0,0,0},{576,1,3,3},{1,1,1,2304},{0,0,0,0},{96,576,1,1},{1,1,1,384},
	{576,96,1,1},{1,1,1,2304},{0,0,0,0},{576,1,3,3},{1,1,1,2304},{0,0,0,0},{96,576,1,1},{1,1,1,384},
	{576,96,1,1},{1,1,1,2304},{0,0,0,0},{576,1,3,3},{1,1,1,2304},{0,0,0,0},{160,576,1,1},{1,1,1,640},
	{960,160,1,1},{1,1,1,3840},{0,0,0,0},{960,1,3,3},{1,1,1,3840},{0,0,0,0},{160,960,1,1},{1,1,1,640},
	{960,160,1,1},{1,1,1,3840},{0,0,0,0},{960,1,3,3},{1,1,1,3840},{0,0,0,0},{160,960,1,1},{1,1,1,640},
	{960,160,1,1},{1,1,1,3840},{0,0,0,0},{960,1,3,3},{1,1,1,3840},{0,0,0,0},{320,960,1,1},{1,1,1,1280},
	{1280,320,1,1},{1,1,1,5120},{0,0,0,0}
};

/*
 * param[layer] = {strides, padding, group}
 * layer 0, 18 : ConvBNReLU
 * layer 1 : ConvBNReLU, Conv, BN
 * layer 2~ 17 : ConvBNReLU, ConvBNReLU, Conv, BN
 */
const int params_[][3] = {
	{2,1,1},{0,0,0},{0,0,0},
	{1,1,32},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{2,1,96},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,144},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{2,1,144},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,192},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,192},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{2,1,192},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,384},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,384},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,384},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,384},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,576},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,576},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{2,1,576},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,960},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,960},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0},{1,1,960},{0,0,0},{0,0,0},{1,0,1},{0,0,0},
	{1,0,1},{0,0,0},{0,0,0}
};

void PrintModel(net* n)
{
	char name[5][20] = {"CONV","BN","RELU", "DROP", "LINEAR"};
	printf("model : %s\n",n->name);
	for(int i=0;i<n->size;i++)
	{
		layer* lptr = &n->layers[i];
		for(int j=0;j<lptr->size;j++)
		{
			sublayer* slptr = &lptr->sublayers[j];
			for(int k=0;k<slptr->size;k++)
			{
				operation* op= &slptr->ops[k];
				if(op->opcode == CONV)
				{
					printf("model[layer:%d][sublayer:%d][op:%d] : %s, shape(%d,%d,%d,%d), param(stride:%d,padding %d,group %d)\n", i,j,k,name[CONV],
						op->filter->out_channel, op->filter->in_channel, op->filter->height, op->filter->width,
						op->param->strides, op->param->padding, op->param->groups);
				}
				else if( op -> opcode == BN)
				{
					printf("model[layer:%d][sublayer:%d][op:%d] : %s, shape(%d,%d,%d,%d)\n", i,j,k,name[BN],
						op->filter->out_channel, op->filter->in_channel, op->filter->height, op->filter->width);

				}
				else if( op -> opcode == RELU)
				{
					printf("model[layer:%d][sublayer:%d][op:%d] : %s\n", i,j,k,name[RELU]);
				}
				else if( op -> opcode == DROP)
				{
					printf("model[layer:%d][sublayer:%d][op:%d] : %s\n", i,j,k,name[DROP]);
				}
				else if( op -> opcode == RELU)
				{
					printf("model[layer:%d][sublayer:%d][op:%d] : %s\n", i,j,k,name[LINEAR]);
				}			
				if(op->filter)
				{
					//PrintMat(op->filter);
				}
			}
		}
	}
}


void ReadBinFile_(DTYPE* data, std::string target)
{

    std::string root = "./Weights/";
    std::string test = root+target + ".bin";
	std::cout<< test << std::endl;
    if( data==NULL)
    {
		printf("Data is NULL\n");
		exit(1);
	}

    int index=0;
    DTYPE load_val;
    std::ifstream read_file(test, std::ios::binary);
    if ( !read_file.is_open() )
    {
        std::cout<<"No Such Binaray"<<std::endl;
        exit(-1);
    }
    while( read_file.read(reinterpret_cast<char*>(&load_val), sizeof(DTYPE)))
    {
        data[index++] = load_val;
    }
}

void ReadWeights(net* n)
{
	const char layer_name[][20] = {
		"_ConvBNRelu/", "_InvertedResidual/"
	};
	const char weight_name[][20] = {
		"0_Conv","1_BatchNorm_mean","1_BatchNorm_var","1_BatchNorm_beta", "1_BatchNorm_gamma",
		"3_Conv","4_BatchNorm_mean","4_BatchNorm_var","4_BatchNorm_beta", "4_BatchNorm_gamma",
		"6_Conv","7_BatchNorm_mean","7_BatchNorm_var","7_BatchNorm_beta", "7_BatchNorm_gamma",
	};
	
	for( int li =0; li<19; li++)
	{
		layer* lptr = &n->layers[li];

		int idx = 0;
		for (int sli = 0; sli<lptr->size;sli++)
		{
			sublayer* slptr = &lptr->sublayers[sli];
			std::string target = "layer_"+std::to_string(li);
			if(li == 0 || li == 18)
				target += layer_name[0];
			else
				target += layer_name[1];
			for(int opi = 0; opi<slptr->size;opi++)
			{
				operation* op = &slptr->ops[opi];
				if(op->opcode == CONV)
				{
					ReadBinFile_(&op->filter->data[0],target+weight_name[idx++]);
				}
				else if(op->opcode == BN)
				{
					int offset = op->filter->width/4;
					for(int i=0;i<4;i++)
					{
						ReadBinFile_(&op->filter->data[i*offset],target+weight_name[idx++]);
					}
				}
			}
		}
	}
}

net GetMobileNetV2_()
{
	net mobilenetv2;
	net* nptr = &mobilenetv2;
	InitNetwork(nptr,"mobilenetv2",20);
	
	int sublayer_i = 0;
	int opcode_i = 0;
	int shape_i = 0;
	int param_i = 0;
	
	// Iterate network's layers
	for(int li=0;li<nptr->size;li++)
	{
		layer* lptr = &nptr->layers[li];
		InitLayer(lptr,layer_sizes[li]);
		
		// Iterate layer's sublayers
		for(int sli = 0; sli < lptr->size; sli++)
		{
			sublayer* slptr = &lptr->sublayers[sli];
			InitSublayer(slptr,sublayer_sizes[sublayer_i++]);
			
			// Iterate sublayer's operation
			for(int opi = 0; opi < slptr->size; opi++)
			{
				operation* op = &slptr->ops[opi];

				op->filter = (ds*)malloc(sizeof(ds));
				op->param = (Param*)malloc(sizeof(Param));

				InitMat(op->filter,shapes_[shape_i++]);
				InitParam(op->param, params_[param_i++]);
				InitOperation(op,opcodes[opcode_i++]);
			}
		}
	}

	ReadWeights(&mobilenetv2);
	//PrintModel(&mobilenetv2);	
	//PrintTemp(&mobilenetv2);
	//PrintTempTemp(&mobilenetv2);

	return mobilenetv2;
}

