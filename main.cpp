#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CONVOLUTION 0
#define POOLING 1
#define RELU 2

typedef float DTYPE;

typedef struct Mat_ {
	DTYPE* data;
	int width;
	int height;
	int channel;
}Mat;

typedef struct Layer_ {
	int type;
	int padding[4];
	int stride[2];
	int dilation[2];
	Mat kernel;
	// brief : initialize layer
	// param : 
	// layer type : conv, pooling, relu
	// paddings(up,down,left,right)
	// stride(horizontal, vertical)
	// dilation(horizontal, vertical)
	// kernel_conf(height, width, channel)
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
	
	layers[0].InitLayer(CONVOLUTION, (int[4]){1,1,1,1},(int[2]){ 1,1 }, (int[2]){ 1,1 }, {NULL,3,3,3});
	layers[1].InitLayer(RELU, (int[4]){1,1,1,1},(int[2]){ 1,1 }, (int[2]){ 1,1 }, {NULL,3,3,3});
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
