#include <iostream>
#include <fstream>
#include <math.h>
#include "include/Convolution.h"
#include "BatchNormalization.h"

int main()
{

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


