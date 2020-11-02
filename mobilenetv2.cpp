#include <iostream>
#include <fstream>
#include <math.h>
#include "include/Convolution.h"

int main()
{

    ds output_data;
    ds input_data;
    ds filter;
    
    InitParameter(&input_data, 1,3,10,10);
    InitParameter(&filter, 5,3,3,3);

    lc feature;
    feature.padding=1;
    feature.strides=1;
    return 0;
}


