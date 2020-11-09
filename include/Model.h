#pragma once
#include <iostream>
#include "DataStruct.h"
#include "Convolution.h"
#include "BatchNormalization.h"
#include "Activation.h"
#include "NetStruct.h"

// # :  20
const int layer_sizes[] = {
	1,3,4,4,4,
	4,4,4,4,4,
	4,4,4,4,4,
	4,4,4,1,1
};

// # : 72
const int sublayer_sizes[] = {
	3,				//0
	3,1,1,			//1
	3,3,1,1,		//2
	3,3,1,1,		//3
	3,3,1,1,		//4
	3,3,1,1,		//5
	3,3,1,1,		//6
	3,3,1,1,		//7
	3,3,1,1,		//8
	3,3,1,1,		//9
	3,3,1,1,		//10
	3,3,1,1,		//11
	3,3,1,1,		//12
	3,3,1,1,		//13
	3,3,1,1,		//14
	3,3,1,1,		//15
	3,3,1,1,		//16
	3,3,1,1,		//17
	3,				//18
	2				//19
};

const int opcodes[141] = {
	CONV,BN,RELU,							// 0
	CONV,BN,RELU, CONV, BN,					// 1
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 2
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 3
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 4
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 5
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 6
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 7
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 8
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 9
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 10
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 11
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 12
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 13
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 14
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 15
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 16
	CONV,BN,RELU, CONV,BN,RELU, CONV, BN,	// 17
	CONV,BN,RELU, 
	DROP,LINEAR
};
    //std::string root = "/home/alpha930/Desktop/CNetProj/Weights/";
const int shapes[] =
	{32, 3, 3, 3,
	32,
	32, 1, 3, 3,
	32,
	16, 32, 1, 1,
	16,
	96, 16, 1, 1,
	96,
	96, 1, 3, 3,
	96,
	24, 96, 1, 1,
	24,
	144, 24, 1, 1,
	144,
	144, 1, 3, 3,
	144,
	24, 144, 1, 1,
	24,
	144, 24, 1, 1,
	144,
	144, 1, 3, 3,
	144,
	32, 144, 1, 1,
	32,
	192, 32, 1, 1,
	192,
	192, 1, 3, 3,
	192,
	32, 192, 1, 1,
	32,
	192, 32, 1, 1,
	192,
	192, 1, 3, 3,
	192,
	32, 192, 1, 1,
	32,
	192, 32, 1, 1,
	192,
	192, 1, 3, 3,
	192,
	64, 192, 1, 1,
	64,
	384, 64, 1, 1,
	384,
	384, 1, 3, 3,
	384,
	64, 384, 1, 1,
	64,
	384, 64, 1, 1,
	384,
	384, 1, 3, 3,
	384,
	64, 384, 1, 1,
	64,
	384, 64, 1, 1,
	384,
	384, 1, 3, 3,
	384,
	64, 384, 1, 1,
	64,
	384, 64, 1, 1,
	384,
	384, 1, 3, 3,
	384,
	96, 384, 1, 1,
	96,
	576, 96, 1, 1,
	576,
	576, 1, 3, 3,
	576,
	96, 576, 1, 1,
	96,
	576, 96, 1, 1,
	576,
	576, 1, 3, 3,
	576,
	96, 576, 1, 1,
	96,
	576, 96, 1, 1,
	576,
	576, 1, 3, 3,
	576,
	160, 576, 1, 1,
	160,
	960, 160, 1, 1,
	960,
	960, 1, 3, 3,
	960,
	160, 960, 1, 1,
	160,
	960, 160, 1, 1,
	960,
	960, 1, 3, 3,
	960,
	160, 960, 1, 1,
	160,
	960, 160, 1, 1,
	960,
	960, 1, 3, 3,
	960,
	320, 960, 1, 1,
	320,
	1280, 320, 1, 1,
	1280,
	-1,-1,-1,-1
};

int params[] = {
	2,1,1,
	1,1,32,
	1,0,1,
	1,0,1,
	2,1,96,
	1,0,1,
	1,0,1,
	1,1,144,
	1,0,1,
	1,0,1,
	2,1,144,
	1,0,1,
	1,0,1,
	1,1,192,
	1,0,1,
	1,0,1,
	1,1,192,
	1,0,1,
	1,0,1,
	2,1,192,
	1,0,1,
	1,0,1,
	1,1,384,
	1,0,1,
	1,0,1,
	1,1,384,
	1,0,1,
	1,0,1,
	1,1,384,
	1,0,1,
	1,0,1,
	1,1,384,
	1,0,1,
	1,0,1,
	1,1,576,
	1,0,1,
	1,0,1,
	1,1,576,
	1,0,1,
	1,0,1,
	2,1,576,
	1,0,1,
	1,0,1,
	1,1,960,
	1,0,1,
	1,0,1,
	1,1,960,
	1,0,1,
	1,0,1,
	1,1,960,
	1,0,1,
	1,0,1
};

void PrintModel(net* n)
{
	char name[5][20] = {"Conv","BN","RELU", "DROP", "LINEAR"};
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
				if(op->op_type == CONV)
				{
					printf("model[layer:%d][sublayer:%d][op:%d] : %s, shape(%d,%d,%d,%d), param(stride:%d,padding %d,group %d)\n", i,j,k,name[CONV],
						op->filter->out_channel, op->filter->in_channel, op->filter->height, op->filter->width,
						op->conv_p->strides, op->conv_p->padding, op->conv_p->groups);
				}
				else if( op -> op_type == BN)
				{
					printf("model[layer:%d][sublayer:%d][op:%d] : %s, shape(%d,%d,%d,%d)\n", i,j,k,name[BN],
						op->filter->out_channel, op->filter->in_channel, op->filter->height, op->filter->width);

				}
				else if( op -> op_type == RELU)
				{
					printf("model[layer:%d][sublayer:%d][op:%d] : %s\n", i,j,k,name[RELU]);
				}
				else if( op -> op_type == DROP)
				{
					printf("model[layer:%d][sublayer:%d][op:%d] : %s\n", i,j,k,name[DROP]);
				}
				else if( op -> op_type == RELU)
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


void ReadBinFile_(D_type* data, std::string target)
{

    std::string root = "/home/dlwjdaud/mobisprj/Weights/";
    std::string test = root+target + ".bin";

    if( data==NULL)
    {
		printf("Data is NULL\n");
		exit(1);
	}

    int index=0;
    D_type load_val;
    std::ifstream read_file(test, std::ios::binary);
    if ( !read_file.is_open() )
    {
        std::cout<<"No Such Binaray"<<std::endl;
        exit(-1);
    }
    while( read_file.read(reinterpret_cast<char*>(&load_val), sizeof(D_type)))
    {
        data[index++] = load_val;
    }
}


void ReadWeights(net* n)
{
	for( int li =0; li<19; li++)
	{
		layer* lptr = &n->layers[li];
		for (int sli =0; sli<lptr->size;sli++)
		{

			printf("%d %d \n",li,sli);
			sublayer* slptr = &lptr->sublayers[sli];
			std::string target = "";
			if( li == 0 || li == 18)
			{
				operation* op;
				int offset;
				if(sli == 0){
					op = &slptr->ops[0];
					target = "layer_"+std::to_string(li)+"_ConvBNRelu/0_Conv";
					ReadBinFile_(&op->filter->data[0], target);

					op = &slptr->ops[1];
					offset = op->filter->width/4;

					target = "layer_"+std::to_string(li)+"_ConvBNRelu/1_BatchNorm_mean";
					ReadBinFile_(&op->filter->data[offset*0], target);

					target = "layer_"+std::to_string(li)+"_ConvBNRelu/1_BatchNorm_var";
					ReadBinFile_(&op->filter->data[offset*1], target);

					target = "layer_"+std::to_string(li)+"_ConvBNRelu/1_BatchNorm_beta";
					ReadBinFile_(&op->filter->data[offset*2], target);

					target = "layer_"+std::to_string(li)+"_ConvBNRelu/1_BatchNorm_gamma";
					ReadBinFile_(&op->filter->data[offset*3], target);
				}
			}
			else if( li == 1)
			{
				operation* op;
				int offset;
				if(sli == 0){
					op = &slptr->ops[0];

					target = "layer_"+std::to_string(li)+"_InvertedResidual/0_Conv";
					ReadBinFile_(&op->filter->data[offset*0], target);

					op = &slptr->ops[1];
					offset = op->filter->width/4;

					target = "layer_"+std::to_string(li)+"_InvertedResidual/1_BatchNorm_mean";
					ReadBinFile_(&op->filter->data[offset*0], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/1_BatchNorm_var";
					ReadBinFile_(&op->filter->data[offset*1], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/1_BatchNorm_beta";
					ReadBinFile_(&op->filter->data[offset*2], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/1_BatchNorm_gamma";
					ReadBinFile_(&op->filter->data[offset*3], target);
				}
				/////////////////////////////////////////////////////////////////////////
				if(sli == 1){
					op = &slptr->ops[0];

					target = "layer_"+std::to_string(li)+"_InvertedResidual/3_Conv";
					ReadBinFile_(&op->filter->data[offset*0], target);
				}
				/////////////////////////////////////////////////////////////////////////
				if(sli == 2){

					op = &slptr->ops[0];
					offset = op->filter->width/4;

					target = "layer_"+std::to_string(li)+"_InvertedResidual/4_BatchNorm_mean";
					ReadBinFile_(&op->filter->data[offset*0], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/4_BatchNorm_var";
					ReadBinFile_(&op->filter->data[offset*1], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/4_BatchNorm_beta";
					ReadBinFile_(&op->filter->data[offset*2], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/4_BatchNorm_gamma";
					ReadBinFile_(&op->filter->data[offset*3], target);
				}
			}   
			else
			{
				operation* op;
				int offset;
				if(sli == 0)
				{
					/////////////////////////////////////////////////////////////////////////	
					op = &slptr->ops[0];
					target = "layer_"+std::to_string(li)+"_InvertedResidual/0_Conv";
					ReadBinFile_(&op->filter->data[offset*0], target);

					op = &slptr->ops[1];
					offset = op->filter->width/4;

					target = "layer_"+std::to_string(li)+"_InvertedResidual/1_BatchNorm_mean";
					ReadBinFile_(&op->filter->data[offset*0], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/1_BatchNorm_var";
					ReadBinFile_(&op->filter->data[offset*1], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/1_BatchNorm_beta";
					ReadBinFile_(&op->filter->data[offset*2], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/1_BatchNorm_gamma";
					ReadBinFile_(&op->filter->data[offset*3], target);
				}
				/////////////////////////////////////////////////////////////////////////
				else if(sli == 1)
				{
					op = &slptr->ops[0];
					offset = op->filter->width/4;


					target = "layer_"+std::to_string(li)+"_InvertedResidual/3_Conv";
					ReadBinFile_(&op->filter->data[offset*0], target);
					
					op = &slptr->ops[1];
					offset = op->filter->width/4;

					target = "layer_"+std::to_string(li)+"_InvertedResidual/4_BatchNorm_mean";
					ReadBinFile_(&op->filter->data[offset*0], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/4_BatchNorm_var";
					ReadBinFile_(&op->filter->data[offset*1], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/4_BatchNorm_beta";
					ReadBinFile_(&op->filter->data[offset*2], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/4_BatchNorm_gamma";
					ReadBinFile_(&op->filter->data[offset*3], target);
				}
				/////////////////////////////////////////////////////////////////////////
				else if(sli == 2){
					op = &slptr->ops[0];

					target = "layer_"+std::to_string(li)+"_InvertedResidual/6_Conv";
					ReadBinFile_(&op->filter->data[offset*0], target);
				}
				/////////////////////////////////////////////////////////////////////////
				else if(sli == 3){
					op = &slptr->ops[0];
					offset = op->filter->width/4;

					target = "layer_"+std::to_string(li)+"_InvertedResidual/7_BatchNorm_mean";
					ReadBinFile_(&op->filter->data[offset*0], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/7_BatchNorm_var";
					ReadBinFile_(&op->filter->data[offset*1], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/7_BatchNorm_beta";
					ReadBinFile_(&op->filter->data[offset*2], target);

					target = "layer_"+std::to_string(li)+"_InvertedResidual/7_BatchNorm_gamma";
					ReadBinFile_(&op->filter->data[offset*3], target);
				}
			}

		}
	}

}
net GetMobileNetV2()
{
	operation* ops = (operation*)malloc(sizeof(operation)*141);
	
	int si = 0;
	int pi = 0;
	for(int i=0;i<141;i++)
	{
		ds* filter;		
		conv_param* param;	
		if(opcodes[i] == CONV)
		{
			filter = (ds*)malloc(sizeof(ds));
			param = (conv_param*)malloc(sizeof(conv_param));

			InitParameter(filter,shapes[si], shapes[si+1], shapes[si+2], shapes[si+3]);
			InitConvParam(param, params[pi], params[pi+1], params[pi+2]);
			si+=4; 
			pi+=3;
		}
		else if(opcodes[i] == BN)
		{
			filter = (ds*)malloc(sizeof(ds));
			param = NULL;
			InitParameter(filter,1, 1, 1, shapes[si]*4);
			si++;
		}
		else if(opcodes[i]  == RELU)
		{
			filter = NULL;
			param = NULL;
		}
		else if(opcodes[i] == DROP)
		{
		}
		else if (opcodes[i] == LINEAR)
		{
		}
		InitOperation(&ops[i], opcodes[i], filter, param);
	}

	sublayer* sublayers = (sublayer*)malloc(sizeof(sublayer)*72);
	int oi = 0;
	for(int i=0;i<72;i++)
	{
		sublayers[i].size = sublayer_sizes[i];
		sublayers[i].ops = (operation*)malloc(sizeof(operation)*sublayer_sizes[i]);
		for(int j=0;j<sublayers[i].size;j++)
		{
			PushOperation(&sublayers[i],&ops[oi++], j);
		}
	}
	
	layer* layers = (layer*)malloc(sizeof(layer)*20);
	int sbi = 0;
	for(int i=0;i<20;i++)
	{
		layers[i].size = layer_sizes[i];
		layers[i].sublayers = (sublayer*)malloc(sizeof(sublayer)*layer_sizes[i]);
		for(int j=0;j<layers[i].size;j++)
		{
			PushSublayer(&layers[i], &sublayers[sbi++],j);
		}
	}
	
	net mobilenetv2;
	strcpy(mobilenetv2.name, "mobilenetv2");
	mobilenetv2.size = 20;
	mobilenetv2.layers = (layer*)malloc(sizeof(layer)*20);
	for(int i=0;i<20;i++){
		PushLayer(&mobilenetv2, &layers[i],i);
	}
	ReadWeights(&mobilenetv2);
	PrintModel(&mobilenetv2);	
	return mobilenetv2;
}



