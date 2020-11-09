#include <iostream>
#include <fstream>
#include <math.h>
#include <assert.h>
#include "include/Convolution.h"
#include "include/BatchNormalization.h"
#include "include/NetStruct.h" 
#include "include/Model.h"

void ReadBinFile(const char* path, D_type* data)
{
	int idx = 0;
	float temp;
	std::cout<< path<<std::endl;
	std::ifstream ifs(path, std::ios::binary);
    while( ifs.read(reinterpret_cast<char*>(&temp), sizeof(D_type)))
	{
		data[idx++] = temp;
	}
}

bool CompareMat(ds* x,ds* y)
{
	int x_size = GetTotalSize(x);
	int y_size = GetTotalSize(y);
	assert(x_size == y_size);

	for(int i = 0; i< x_size ; i++)
    {  
		//printf("%f | %f\n",x->data[i], y->data[i]);
		if( std::abs( x->data[i] - y->data[i] ) > 0.0001)
        {
            std::cout.precision(10);

            std::cout<<"ERROR "<< x->data[i]<<" | "<< y->data[i]<<std::endl;
            std::cout<<"IDX : "<<i<<std::endl;
			return false;
        }
    }
	return true;
}

void BatchNormalizationTest()
{
	ds input;
	ds filter;
	ds v_output;
	ds my_output;

	InitParameter(&input, 1, 32, 112, 112);
	InitParameter(&filter, 1, 1, 1, 4*32);
	InitParameter(&v_output, 1, 32, 112, 112);

	ReadBinFile("./validation_data/input_data.bin",input.data);
	ReadBinFile("./validation_data/output_data.bin",v_output.data);
	ReadBinFile("./validation_data/moving_mean.bin",&filter.data[0]);
	ReadBinFile("./validation_data/moving_var.bin",&filter.data[32]);
	ReadBinFile("./validation_data/beta.bin",&filter.data[64]);
	ReadBinFile("./validation_data/gamma.bin",&filter.data[96]);
	
	//BatchNormalization(&input, &filter, &my_output, NULL);
	my_output = BatchNormalization_(&input, &filter, NULL);
	
	// validation test
	if(CompareMat(&v_output, &my_output) == false)
		std::cout<<"Validation fail"<<std::endl;
	else
		std::cout<<"Validation Success"<<std::endl;
    
}

void ConvolutionTest()
{
	ds input;
	ds v_output;
	ds my_output;
	ds filter;
	conv_param conv_p;
	InitConvParam(&conv_p, 1, 1, 1);
 
	InitParameter(&input, 1, 32, 112, 112);
	InitParameter(&filter, 32, 32, 3, 3);
	InitParameter(&v_output, 1, 32, 112, 112);
	
	ReadBinFile("./validation_data/conv_in.bin",input.data);
	ReadBinFile("./validation_data/conv_fil.bin",filter.data);
	ReadBinFile("./validation_data/conv_out.bin",v_output.data);
	
	//NaiveConvolution(&input, &filter, &my_output, &conv_p);
	my_output = Convolution_(&input, &filter, &conv_p);

	// validation test
	if(CompareMat(&v_output, &my_output) == false)
		std::cout<<"Validation fail"<<std::endl;
	else
		std::cout<<"Validation Success"<<std::endl;

    
}


void BlockTest()
{
	int sublayer_size = 3;
	operation* ops = (operation*)malloc(sizeof(operation)*sublayer_size);
	
	ds input;
	ds output;
	ds v_output;
	ds conv_filter;
	ds bn_filter;
	conv_param conv_p;

	InitConvParam(&conv_p, 2, 1, 1 );
	InitParameter(&input,1,3,224,224 );
	InitParameter(&v_output, 1, 32, 112, 112);
	InitParameter(&conv_filter, 32, 3, 3, 3);
	InitParameter(&bn_filter, 1, 1, 1, 32*4);
	ReadBinFile("./Param/input.bin",input.data);
	ReadBinFile("./Param/output.bin",v_output.data);
	ReadBinFile("./Param/0.0.weight.bin",conv_filter.data);
	ReadBinFile("./Param/moving_mean.bin",&bn_filter.data[0]);
	ReadBinFile("./Param/moving_var.bin",&bn_filter.data[32]);
	ReadBinFile("./Param/beta.bin",&bn_filter.data[64]);
	ReadBinFile("./Param/gamma.bin",&bn_filter.data[96]);


	InitOperation(&ops[0], CONV, &conv_filter, &conv_p);
	InitOperation(&ops[1], BN, &bn_filter, NULL);
	InitOperation(&ops[2], RELU, NULL, NULL);

	sublayer sl;
	InitSublayer(&sl, sublayer_size);
	for(int i = 0; i< sublayer_size; i++)
	{
		PushOperation(&sl, &ops[i], i);
	}
	output = ForwardSublayer(&sl, &input);

	//validation test
	if(CompareMat(&v_output, &output) == false)
		std::cout<<"Validation fail"<<std::endl;
	else
		std::cout<<"Validation Success"<<std::endl;
}

int main()
{

	std::cout<<"TEST"<<std::endl;
	//BatchNormalizationTest();	
	//ConvolutionTest();
	//BlockTest();
	//return 0;
	net network = GetMobileNetV2();
	ds input;
	ds output;
	ds v_output;
	InitParameter(&input, 1, 3, 224, 224);
	ReadBinFile("./Weights/input/input_data.bin",input.data);

	//InitParameter(&v_output, 1, 16, 112, 112);	//0
	//InitParameter(&v_output, 1, 16, 112, 112);	//1
	//InitParameter(&v_output, 1, 24, 56, 56);		//2
	//InitParameter(&v_output, 1, 24, 56, 56);		//3
	//InitParameter(&v_output, 1, 32, 28, 28);	//4
	InitParameter(&v_output, 1, 1280, 7, 7);	//18
	//ReadBinFile("./Param/input.bin",input.data);
	//ReadBinFile("./Param/output.bin",v_output.data);
	
	//ReadBinFile("./Weights/layer_0_ConvBNRelu/imm_out.bin",v_output.data);	//0
	//ReadBinFile("./Weights/layer_1_InvertedResidual/imm_out.bin",v_output.data);	//1
	//ReadBinFile("./Weights/layer_2_InvertedResidual/imm_out.bin",v_output.data);	//2
	//ReadBinFile("./Weights/layer_3_InvertedResidual/imm_out.bin",v_output.data);	//3
	//ReadBinFile("./Weights/layer_4_InvertedResidual/imm_out.bin",v_output.data);	//3
	
	ReadBinFile("./Weights/layer_18_ConvBNRelu/imm_out.bin",v_output.data);		//18
	output = Inference(&network,&input);
	if(CompareMat(&v_output, &output) == false)
		std::cout<<"Validation fail"<<std::endl;
	else
		std::cout<<"Validation Success"<<std::endl;

	return 0;
	ds output_data;
    ds input_data;
    ds filter;
    
    InitParameter(&input_data, 1,1,3,3);
    InitParameter(&filter, 1,1,3,3);
    conv_param feature;
    feature.padding=1;
    feature.strides=1;

    Convolution(&input_data, &filter, &output_data, &feature);


    D_type test_val;
    int tag=0;
	// gamma.bin (32)
	// beta.bin
	// moving.bin

	
    std::ifstream testing("valid.bin", std::ios::binary);
    while( testing.read(reinterpret_cast<char*>(&test_val), sizeof(D_type)))
    {   
        if( std::abs( test_val- output_data.data[tag] ) > 0.0000001)
        {
            std::cout.precision(10);

            std::cout<<"ERROR"<<std::endl;
            std::cout<<"TAG : "<<tag<<std::endl;
            std::cout<<test_val<<std::endl;
            std::cout<< output_data.data[tag] <<std::endl;

            std::cout<< test_val- output_data.data[tag] <<std::endl;
            exit(-1);
        }
        else
        {
        }
        tag++;
    }   
    std::cout<<"Validation Sucess"<<std::endl;
    return 0;

}


