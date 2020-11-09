#include <iostream>
#include <fstream>
#include <math.h>
#include <assert.h>
#include "include/Convolution.h"
#include "include/BatchNormalization.h"
#include "include/NetStruct.h" 
#include "include/Model.h"

void ReadBinFile(const char* path, DTYPE* data)
{
	int idx = 0;
	float temp;
	std::cout<< path<<std::endl;
	std::ifstream ifs(path, std::ios::binary);
    while( ifs.read(reinterpret_cast<char*>(&temp), sizeof(DTYPE)))
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
		if( std::abs( x->data[i] - y->data[i] ) > 0.00002)
        {
            std::cout.precision(10);

            std::cout<<"ERROR "<< x->data[i]<<" | "<< y->data[i]<<std::endl;
            std::cout<<"IDX : "<<i<<std::endl;
			return false;
        }
    }
	return true;
}

int main()
{

	std::cout<<"TEST"<<std::endl;
	net network = GetMobileNetV2_();
	ds input;
	ds output;
	ds v_output;
	InitMat(&input, {1, 3, 224, 224});
	ReadBinFile("./Weights/input/input_data.bin",input.data);

	//InitParameter(&v_output, 1, 16, 112, 112);	//0
	//InitParameter(&v_output, 1, 16, 112, 112);	//1
	//InitParameter(&v_output, 1, 24, 56, 56);		//2
	//InitParameter(&v_output, 1, 24, 56, 56);		//3
	//InitParameter(&v_output, 1, 32, 28, 28);		//4
	InitMat(&v_output, {1, 1280, 7, 7});	//18
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
}


