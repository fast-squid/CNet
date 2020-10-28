#include <iostream>
#include <fstream>
#include <math.h>
#include "include/conv_layer.h"


int main()
{
    
    ds output_data;
    ds input_data;
    ds filter;
    
    InitParamter(&input_data, 1,3,6,6, false);
    InitParamter(&filter, 5,3,3,3, false);

    layer config;
    int pad=3;
    int str=1;
    int dil=1;
    layerInit(&config,pad,str,dil);


    Convolution(&input_data, &filter, &output_data, &config, 0);

    /// Validation
    D_type test_val;
    int tag=0;
    std::ifstream testing("valid.bin", std::ios::binary);
    while( testing.read(reinterpret_cast<char*>(&test_val), sizeof(D_type)))
    {   
        if( std::abs( test_val- output_data.data[tag] ) > 0.00001)
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

}
