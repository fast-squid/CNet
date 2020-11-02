#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define CONVOLUTION 0
#define POOLING 1
#define RELU 2
#define BOTTLENECK 3

typedef float DTYPE;

typedef struct Mat_ 
{
	DTYPE* data;
	int width;
	int height;
	int channel;
	int num;
	void PrintMat(){
		for(int n =0;n<num;n++)
		{
			for(int c=0;c<channel;c++)
			{
				for(int h=0;h<height;h++)
				{
					for(int w=0;w<width;w++)
					{
						int index = n*channel*height*width
							+c*height*width+h*width + w;
						printf("%f ");
					}
					printf("\n");
				}
				printf("\n");
			}
		}
	}
	struct Mat_ PadMatrix(const int (&padding)[4]){
		struct Mat_ p_mat;
		p_mat.num = num;
		p_mat.channel =channel;
		p_mat.height = height + padding[0] + padding[1];
		p_mat.width = width + padding[2] + padding[3];
		p_mat.data = (DTYPE*)calloc(p_mat.num*p_mat.channel*p_mat.height*p_mat.width, sizeof(DTYPE));
		
		for(int n = 0;n < p_mat.num;n++)
		{
			for(int c = 0;c < p_mat.channel;c++)
			{
				for(int h = 0;h < p_mat.height;h++)
				{
					if(h==0 || h== p_mat.height-1)
					{
						continue;
					}
					int p_offset = n*p_mat.channel*p_mat.height*p_mat.width
						+ c*p_mat.height*p_mat.width
						+ h*p_mat.width
						+ padding[2];
					int offset = n*channel*height*width
						+ c*height*width
						+ h*width;
					memcpy(&p_mat.data[p_offset],&data[offset],sizeof(DTYPE)*width);
				}
			}
		}


	}
}Mat;

typedef struct Layer_ {
	int type;
	int padding[4];
	int stride[2];
	int dilation[2];
	Mat kernel;
	// brief : initialize layer
	// params : 
	// layer type : conv, pooling, relu
	// paddings(up,down,left,right)
	// stride(horizontal, vertical)
	// dilation(horizontal, vertical)
	// kernel_conf(height, width, channel)
	// return : none
	void InitLayer(int type_, const int (&padding_)[4],const int (&stride_)[2], const int (&dilation_)[2], Mat kernel_)
	{
		type = type_;
		padding[0] = padding_[0];
		padding[1] = padding_[1];
		padding[2] = padding_[2];
		padding[3] = padding_[3];

		stride[0] = stride_[0];
		stride[1] = stride_[1];

		dilation[0] = dilation_[0];
		dilation[1] = dilation_[1];
		
		kernel = kernel_;
	}
	// brief : forward function
	// params : none
	// return : mat(feature map)
	void Forward() 
	{
		//switch (type) {
			//case CONVOLUTION: ConvForward();
			//case POOLING: PoolingForward();
			//case RELU: ReLUForward();
		//}
	}
}Layer;

typedef struct Net_ {
	int layer_num;
	Layer* layers;
	void InitNetwork(int layer_num_) {
		layer_num = layer_num_;
		layers = (Layer*)calloc(layer_num,sizeof(Layer));
	}
	void SetLayer(Layer layer,int layer_idx)
	{
		layers[layer_idx]= layer;
		printf("layer[%d] : %d\n", layer_idx,layers[layer_idx].type);
	}
}Net;


int main() 
{
	Net net;
	Mat kernel;
	// Load Model

	net.InitNetwork(2);	// MobileNetV2 configuration
	Layer* layers = (Layer*)calloc(net.layer_num, sizeof(Layer));
	
	layers[0].InitLayer(CONVOLUTION, (int[4]){1,1,1,1},(int[2]){ 1,1 }, (int[2]){ 1,1 }, {NULL,3,3,3,3});
	
//	layers[0].InitLayer(CONVOLUTION, (int[4]){1,1,1,1},(int[2]){ 1,1 }, (int[2]){ 1,1 }, {NULL,3,3,3});
//	layers[0].InitLayer(CONVOLUTION, (int[4]){1,1,1,1},(int[2]){ 1,1 }, (int[2]){ 1,1 }, {NULL,3,3,3});
//	layers[0].InitLayer(CONVOLUTION, (int[4]){1,1,1,1},(int[2]){ 1,1 }, (int[2]){ 1,1 }, {NULL,3,3,3});
//	layers[0].InitLayer(CONVOLUTION, (int[4]){1,1,1,1},(int[2]){ 1,1 }, (int[2]){ 1,1 }, {NULL,3,3,3});
//	layers[0].InitLayer(CONVOLUTION, (int[4]){1,1,1,1},(int[2]){ 1,1 }, (int[2]){ 1,1 }, {NULL,3,3,3});
//	layers[0].InitLayer(CONVOLUTION, (int[4]){1,1,1,1},(int[2]){ 1,1 }, (int[2]){ 1,1 }, {NULL,3,3,3});
//	layers[0].InitLayer(CONVOLUTION, (int[4]){1,1,1,1},(int[2]){ 1,1 }, (int[2]){ 1,1 }, {NULL,3,3,3});
//	layers[0].InitLayer(CONVOLUTION, (int[4]){1,1,1,1},(int[2]){ 1,1 }, (int[2]){ 1,1 }, {NULL,3,3,3});

	for (int i = 0; i < net.layer_num; i++) {
		net.SetLayer(layers[i], i);
	}

	return 0;
}
