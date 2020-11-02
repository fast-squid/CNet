import torch
import torch.nn as nn
import torchvision.models as models
import pickle


def GetParam():
    mv2 = models.mobilenet_v2(pretrained=True)
    mv2.parameters()

    data = []
    for name, param in mv2.named_parameters():
        data.append({"name":name, 'data':param.data.numpy()})

    with open("./mobilenet_v2.pkl", 'wb') as f:
        pickle.dump(data,f)
    with open("./mobilenet_v2.pkl","rb") as f:
        load_data = pickle.load(f)

def CutLayer():
    #### Cut Layer
    mv2_cut = models.mobilenet_v2(pretrained=True).features[:1]
    print(mv2_cut)
    for name, param in mv2_cut.named_parameters():
        print(name)

def SimpleRunning():

    #### TEST
    mv2_cut.eval()
    data = torch.randn(1,3,224,224)
    output = mv2_cut(data)
    print(output.shape)

    ############################ CUSTOM #################################
    arr=[1,2,3,4,5,6,7,8,9]

    i = torch.tensor(arr)
    i = torch.reshape( i, (1,1,3,3))
    w = torch.tensor(arr)
    w= torch.reshape( w, (1,1,3,3))

    conv = torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    conv.weight.data = w
    model = conv.eval()

    output = model(i)
    print(output)

def TESTlayer():

    CutLayer()
    model = nn.Sequential(

    )
    )
    return


TESTlayer()
