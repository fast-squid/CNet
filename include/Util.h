#pragma once
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "DataStruct.h"


void InitWeight(ds* data, std::string target)
{
    std::string root = "/home/alpha930/Desktop/CNetProj/Weights/";
    std::string test = root+target + ".bin";
    int total_ele_size = data->out_channel*data->in_channel*data->height*data->width;

    if( data->data==NULL)
    {
        data->data = (D_type*)malloc(sizeof(D_type)*total_ele_size);
    }

    int index=0;
    D_type load_val;
    std::ifstream read_file(test, std::ios::binary);
    if ( !read_file.is_open() )
    {
        std::cout<<"No Such Binaray"<<std::endl;
        exit(-1);
    }
    while( read_file.read(reinterpret_cast<char*>(&load_val), sizeof(D_type)))
    {
        data->data[index++] = load_val;
        if ( index > total_ele_size )
        {
            std::cerr<<"Allocation ERROR "<<std::endl;
            exit(-1);
        }
    }
}


void Printer(ds layer)
{
    std::cout<<"( "<<layer.out_channel<<" , "<< layer.in_channel<<" , "<< layer.height<< " , "<<layer.width<<" )"<<std::endl;
}

void InitAllParam(lw* layer_pack)
{
    std::string root = "/home/alpha930/Desktop/CNetProj/Weights/";
    int layer_config[] =
{32, 3, 3, 3,
32,
32, 1, 3, 3,
32,
16, 32, 1, 1,
16,
96, 16, 1, 1,
96,
96, 1, 3, 3,
96,
24, 96, 1, 1,
24,
144, 24, 1, 1,
144,
144, 1, 3, 3,
144,
24, 144, 1, 1,
24,
144, 24, 1, 1,
144,
144, 1, 3, 3,
144,
32, 144, 1, 1,
32,
192, 32, 1, 1,
192,
192, 1, 3, 3,
192,
32, 192, 1, 1,
32,
192, 32, 1, 1,
192,
192, 1, 3, 3,
192,
32, 192, 1, 1,
32,
192, 32, 1, 1,
192,
192, 1, 3, 3,
192,
64, 192, 1, 1,
64,
384, 64, 1, 1,
384,
384, 1, 3, 3,
384,
64, 384, 1, 1,
64,
384, 64, 1, 1,
384,
384, 1, 3, 3,
384,
64, 384, 1, 1,
64,
384, 64, 1, 1,
384,
384, 1, 3, 3,
384,
64, 384, 1, 1,
64,
384, 64, 1, 1,
384,
384, 1, 3, 3,
384,
96, 384, 1, 1,
96,
576, 96, 1, 1,
576,
576, 1, 3, 3,
576,
96, 576, 1, 1,
96,
576, 96, 1, 1,
576,
576, 1, 3, 3,
576,
96, 576, 1, 1,
96,
576, 96, 1, 1,
576,
576, 1, 3, 3,
576,
160, 576, 1, 1,
160,
960, 160, 1, 1,
960,
960, 1, 3, 3,
960,
160, 960, 1, 1,
160,
960, 160, 1, 1,
960,
960, 1, 3, 3,
960,
160, 960, 1, 1,
160,
960, 160, 1, 1,
960,
960, 1, 3, 3,
960,
320, 960, 1, 1,
320,
1280, 320, 1, 1,
1280,
-1,-1,-1,-1
};

    int indexer=0;
    //layer_pack = (lw*)malloc(sizeof(lw)*52);
    for( int i =0; i<52; i++)
    {
   
        layer_pack[i].weight.data=NULL;
        layer_pack[i].weight.out_channel=layer_config[indexer++];
        layer_pack[i].weight.in_channel=layer_config[indexer++];
        layer_pack[i].weight.height= layer_config[indexer++];
        layer_pack[i].weight.width = layer_config[indexer++];

        layer_pack[i].mean.data=NULL;
        layer_pack[i].mean.out_channel=1;
        layer_pack[i].mean.in_channel=layer_config[indexer];
        layer_pack[i].mean.height= 1;
        layer_pack[i].mean.width = 1;

        layer_pack[i].var.data=NULL;
        layer_pack[i].var.out_channel=1;
        layer_pack[i].var.in_channel=layer_config[indexer];
        layer_pack[i].var.height= 1;
        layer_pack[i].var.width = 1;

        layer_pack[i].gamma.data=NULL;
        layer_pack[i].gamma.out_channel=1;
        layer_pack[i].gamma.in_channel=layer_config[indexer];
        layer_pack[i].gamma.height= 1;
        layer_pack[i].gamma.width = 1;

        layer_pack[i].beta.data=NULL;
        layer_pack[i].beta.out_channel=1;
        layer_pack[i].beta.in_channel=layer_config[indexer++];
        layer_pack[i].beta.height= 1;
        layer_pack[i].beta.width = 1;

        /*
        Printer(layer_pack[i].weight);
        Printer(layer_pack[i].mean);
        Printer(layer_pack[i].var);
        Printer(layer_pack[i].gamma);
        Printer(layer_pack[i].beta);
        */
    }

    indexer=0;
    for( int block =0; block<19; block++)
    {
        std::string target = "";
        if( block == 0 || block == 18)
        {
            target = "layer_"+std::to_string(block)+"_ConvBNRelu/0_Conv";
            InitWeight(&layer_pack[indexer].weight, target);

            target = "layer_"+std::to_string(block)+"_ConvBNRelu/1_BatchNorm_mean";
            InitWeight(&layer_pack[indexer].mean, target);

            target = "layer_"+std::to_string(block)+"_ConvBNRelu/1_BatchNorm_var";
            InitWeight(&layer_pack[indexer].var, target);

            target = "layer_"+std::to_string(block)+"_ConvBNRelu/1_BatchNorm_gamma";
            InitWeight(&layer_pack[indexer].gamma, target);

            target = "layer_"+std::to_string(block)+"_ConvBNRelu/1_BatchNorm_beta";
            InitWeight(&layer_pack[indexer].beta, target);
            indexer++;
        }
        else
        {
            if( block == 1)
            {
                target = "layer_"+std::to_string(block)+"_InvertedResidual/0_Conv";
                InitWeight(&layer_pack[indexer].weight, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/1_BatchNorm_mean";
                InitWeight(&layer_pack[indexer].mean, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/1_BatchNorm_var";
                InitWeight(&layer_pack[indexer].var, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/1_BatchNorm_gamma";
                InitWeight(&layer_pack[indexer].gamma, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/1_BatchNorm_beta";
                InitWeight(&layer_pack[indexer].beta, target);
                indexer ++;

                target = "layer_"+std::to_string(block)+"_InvertedResidual/3_Conv";
                InitWeight(&layer_pack[indexer].weight, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/4_BatchNorm_mean";
                InitWeight(&layer_pack[indexer].mean, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/4_BatchNorm_var";
                InitWeight(&layer_pack[indexer].var, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/4_BatchNorm_gamma";
                InitWeight(&layer_pack[indexer].gamma, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/4_BatchNorm_beta";
                InitWeight(&layer_pack[indexer].beta, target);
                indexer ++;
            }   
            else
            {
                target = "layer_"+std::to_string(block)+"_InvertedResidual/0_Conv";
                InitWeight(&layer_pack[indexer].weight, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/1_BatchNorm_mean";
                InitWeight(&layer_pack[indexer].mean, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/1_BatchNorm_var";
                InitWeight(&layer_pack[indexer].var, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/1_BatchNorm_gamma";
                InitWeight(&layer_pack[indexer].gamma, target);

                target = "layer_"+std::to_string(block)+"_InvertedResidual/1_BatchNorm_beta";
                InitWeight(&layer_pack[indexer].beta, target);
                indexer ++;
                target = "layer_"+std::to_string(block)+"_InvertedResidual/3_Conv";
                InitWeight(&layer_pack[indexer].weight, target);
                target = "layer_"+std::to_string(block)+"_InvertedResidual/4_BatchNorm_mean";
                InitWeight(&layer_pack[indexer].mean, target);
                target = "layer_"+std::to_string(block)+"_InvertedResidual/4_BatchNorm_var";
                InitWeight(&layer_pack[indexer].var, target);
                target = "layer_"+std::to_string(block)+"_InvertedResidual/4_BatchNorm_gamma";
                InitWeight(&layer_pack[indexer].gamma, target);
                target = "layer_"+std::to_string(block)+"_InvertedResidual/4_BatchNorm_beta";
                InitWeight(&layer_pack[indexer].beta, target);
                indexer ++;
                target = "layer_"+std::to_string(block)+"_InvertedResidual/6_Conv";
                InitWeight(&layer_pack[indexer].weight, target);
                target = "layer_"+std::to_string(block)+"_InvertedResidual/7_BatchNorm_mean";
                InitWeight(&layer_pack[indexer].mean, target);
                target = "layer_"+std::to_string(block)+"_InvertedResidual/7_BatchNorm_var";
                InitWeight(&layer_pack[indexer].var, target);
                target = "layer_"+std::to_string(block)+"_InvertedResidual/7_BatchNorm_gamma";
                InitWeight(&layer_pack[indexer].gamma, target);
                target = "layer_"+std::to_string(block)+"_InvertedResidual/7_BatchNorm_beta";
                InitWeight(&layer_pack[indexer].beta, target);
                indexer ++;
            }
        }
    }
    return;
}
