#include <iostream>
#include <fstream>
#include <math.h>
#include "include/Convolution.h"
#include "BatchNormalization.h"

#define EPSILON 0.000001

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
	ds v_output;
	ds my_output;

	InitParameter(&input, 1, 32, 112, 112);
	InitParameter(&v_output, 1, 32, 112, 112);

	D_type gamma[32];
	D_type beta[32];
	D_type moving_mean[32];
	D_type moving_var[32];
	D_type eps = EPSILON;

	ReadBinFile("./validation_data/input_data.bin",input.data);
	ReadBinFile("./validation_data/output_data.bin",v_output.data);
	ReadBinFile("./validation_data/moving_mean.bin",moving_mean);
	ReadBinFile("./validation_data/moving_var.bin",moving_var);
	ReadBinFile("./validation_data/beta.bin",beta);
	ReadBinFile("./validation_data/gamma.bin",gamma);
	
	BatchNorm(&input, gamma, beta, eps, moving_mean, moving_var, &my_output);
	// validation test

	for(int i = 0; i< 1*32*112*112 ; i++)
    {   
        if( std::abs( v_output.data[i] - my_output.data[i] ) > 0.000001)
        {
            std::cout.precision(10);

            std::cout<<"ERROR "<< v_output.data[i]<<" | "<<my_output.data[i]<<std::endl;
            std::cout<<"IDX : "<<i<<std::endl;
        }
    }
	std::cout<<"final Validation Success"<<std::endl;
    
}

void GroupConvolutionTest()
{
	ds input;
	ds v_output;
	ds my_output;
	ds filter;
	lc layer;
	layerInit(&layer, 1, 1, 2);
 
	InitParameter(&input, 1, 32, 112, 112);
	InitParameter(&filter, 32, 16, 3, 3);
	InitParameter(&v_output, 1, 32, 112, 112);
	
	ReadBinFile("./validation_data/a.bin",input.data);
	ReadBinFile("./validation_data/b.bin",filter.data);
	ReadBinFile("./validation_data/c.bin",v_output.data);

	GroupConvolution(&input, &filter, &my_output, &layer);
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


int main()
{
	//BatchNormalizationTest();
	GroupConvolutionTest();
    return 0;
	ds output_data;
    ds input_data;
    ds filter;
    
    InitParameter(&input_data, 1,1,3,3);
    InitParameter(&filter, 1,1,3,3);
    lc feature;
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


