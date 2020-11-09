#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"
#include <string.h>

ds Skipconnection(ds* input, ds* output)
{
	ds new_output;
	new_output.out_channel = input->out_channel;
	new_output.in_channel = input->in_channel;
	new_output.height = input->height;
	new_output.width = input->width;
	new_output.data = (D_type*)malloc(sizeof(D_type)*GetTotalSize(input));
	for(int i=0;i<GetTotalSize(input);i++)
	{
		new_output.data[i] =  output->data[i] + input->data[i];
	}
	std::cout<<"\t\tSkipconnection done"<<std::endl;
    return new_output;
}


