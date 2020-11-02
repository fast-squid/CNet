#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct Mat_
{
	float* data;
	char name[100];
	int num;
}Mat;


void WriteModel(Mat* weights)
{
	FILE *fp = fopen("MobileNetV2_model.bin","wb");
	for(int i=0;i<262;i++)
	{
		fwrite(weights[i].data,sizeof(float),weights[i].num,fp);
	}
	fclose(fp);
}
void WriteConfig(Mat* weights)
{
	FILE* fp = fopen("MobileNetV2_model.shape","w");
	for(int i=0;i<262;i++)
	{
		fprintf(fp,"%s %d\n",weights[i].name, weights[i].num);
	}
	fclose(fp);

}


Mat* LoadModel()
{
	FILE *fp_config = fopen("MobileNetV2_model.shape","r");
	int kernel_num;
	
	fscanf(fp_config,"%d",&kernel_num);
	
	Mat* weights = (Mat*)malloc(sizeof(Mat)*kernel_num);
	int i = 0;
	while(fscanf(fp_config,"%s %d",weights[i].name, &weights[i].num)!=EOF)
	{
		weights[i].data = (float*)malloc(sizeof(float)*weights[i].num);
		i++;
	}
	fclose(fp_config);	
	
	// load MobileNetV2 weights
	FILE *fp = fopen("MobileNetV2_model.bin","rb");
	for(int i=0;i<kernel_num;i++)
	{
		fread(weights[i].data,sizeof(float),weights[i].num,fp);
	}
	fclose(fp);

	return weights;
}
int main()
{
	//WriteModel(LoadModel());
	//Mat* weights = LoadModel_bin();
	//WriteConfig(LoadModel());
	Mat* weights = LoadModel();
	printf("%s %d\n",weights[261].name, weights[261].num);
	for(int i =0 ; i<weights[261].num;i++)
	{
		printf("%d %f\n",i, weights[261].data[i]);
	}
	return 0;
}
