#pragma once
typedef  float DTYPE;

typedef struct weight_data_struct
{
    DTYPE* data=NULL;
    int out_channel = 0; // # of Filter
    int in_channel = 0;  
    int height = 0;
    int width = 0;
} ds;

typedef struct Param_
{ 
    int strides=1;
	int padding=1;
    int groups=1;
}Param;

inline int GetTotalSize(ds* mat)
{
	return mat->out_channel * mat->in_channel * mat->height * mat->width; 
}

inline int GetMatSize(ds* mat)
{
	return mat->height * mat->width; 
}

void InitMat(ds* data, const int (&shape)[4])
{
    data->out_channel = shape[0];
    data->in_channel = shape[1];
    data->height = shape[2];
    data->width = shape[3];
	int total_size =GetTotalSize(data);
	
	if(total_size)
	    data->data = (DTYPE*)malloc(sizeof(DTYPE)*total_size);
	else
	{
		free(data);
	}
}

void InitParam(Param* conv_p, const int (&param)[3])
{
    conv_p->strides = param[0];
	conv_p->padding = param[1];
    conv_p->groups = param[2];
}



