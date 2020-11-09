#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"
#include <string.h>

ds Skipconnection(ds* input, ds* output)
{
	ds new_output;
	InitMat(&new_output,{input->out_channel, input->in_channel, input->height, input->width});
	for(int i=0;i<GetTotalSize(input);i++)
	{
		new_output.data[i] =  output->data[i] + input->data[i];
	}
	std::cout<<"\t\t\t\t\t\tSkipconnection done"<<std::endl;
    return new_output;
}


