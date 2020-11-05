#include <iostream>
#include <fstream>
#include <math.h>
#include "include/Convolution.h"
#include "include/BatchNormalization.h"
#include "include/Block.h" 

void ReadBinFile(char* path, D_type* data)
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
	
	BatchNormalization(&input, &filter, &my_output, NULL);
	// validation test

	for(int i = 0; i< 1*32*112*112 ; i++)
    {   
        if( std::abs( v_output.data[i] - my_output.data[i] ) > 0.00001)
        {
            std::cout.precision(10);

            std::cout<<"ERROR "<< v_output.data[i]<<" | "<<my_output.data[i]<<std::endl;
            std::cout<<"IDX : "<<i<<std::endl;
        }
    }
	std::cout<<"final Validation Success"<<std::endl;
    
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
	
	ReadBinFile("./validation_data/a.bin",input.data);
	ReadBinFile("./validation_data/b.bin",filter.data);
	ReadBinFile("./validation_data/c.bin",v_output.data);

	NaiveConvolution(&input, &filter, &my_output, &conv_p);
	// validation test

	for(int i = 0; i< 1*32*112*112 ; i++)
    {   
        if( std::abs( v_output.data[i] - my_output.data[i] ) > 0.0001)
        {
            std::cout.precision(10);

            std::cout<<"ERROR "<< v_output.data[i]<<" | "<<my_output.data[i]<<std::endl;
            std::cout<<"IDX : "<<i<<std::endl;
        }
    }
	std::cout<<"final Validation Success"<<std::endl;
    
}

void BlockTest()
{
	int block_size = 3;
	sublayer* sl = (sublayer*)malloc(sizeof(sublayer)*block_size);
	
	ds input;
	ds filter;
	ds output;
	ds v_output;

	conv_param conv_p;
	InitConvParam(&conv_p, 1, 1, 1);
	InitParameter(&input, 1, 32, 112, 112);
	InitParameter(&filter, 32, 32, 3, 3);
	InitParameter(&v_output, 1, 32, 112, 112);
	
	ReadBinFile("./validation_data/a.bin",input.data);
	ReadBinFile("./validation_data/b.bin",filter.data);
	ReadBinFile("./validation_data/c.bin",v_output.data);


	InitSubLayer(&sl[0], CONV);
	InitSubLayer(&sl[1], BN);
	InitSubLayer(&sl[2], RELU);
	block blk;
	InitBlock(&blk, block_size);
	for(int i = 0; i< block_size; i++)
	{
		PushSubLayer(&blk, &sl[i], i);
	}
	ForwardBlock(&blk, &input, &filter, &output, &conv_p);
}

int main()
{

	std::cout<<"TEST"<<std::endl;
	//BatchNormalizationTest();
	//ConvolutionTest();
	BlockTest();
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


