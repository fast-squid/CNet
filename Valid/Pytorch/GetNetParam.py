import torch
import torch.nn as nn
import torchvision.models as models
import pickle
import numpy as np
import math

alpha_root = "/home/alpha930/Desktop/CNetProject/Param/"
squid_root = "/home/dlwjdaud/mobisprj/Valid/Pytorch/validation_data/"

def CMP(t1,t2,bound=0.00001):
    if np.all( np.abs(t1-t2)< bound ):
        return True
    else:
        return False

def Tensorize(data):
    tensor = torch.tensor(data)
    return tensor

def Numpyize(data):
    nump = data.data.numpy()
    return nump


def GetParam(get_out=False):
    mv2 = models.mobilenet_v2(pretrained=True)
    mv2 = mv2.eval()
    #v2.parameters()

    data = []
    for name, param in mv2.named_parameters():
        data.append({"name":name, 'data':param.data.numpy()})
        param.data.numpy().astype("float32").tofile(squid_root+str(name)+".bin")
    
    if get_out:
        with open("./mobilenet_v2.pkl", 'wb') as f:
            pickle.dump(data,f)
        with open("./mobilenet_v2.pkl","rb") as f:
            load_data = pickle.load(f)
    
    return data



def CutLayer(start_p, end_p, debug=False):
    #### Cut Layer
    data = []
    mv2_cut = models.mobilenet_v2(pretrained=True).features[start_p:end_p]
    mv2_cut = mv2_cut.eval()

    if debug:
        print(mv2_cut)
    for name, param in mv2_cut.named_parameters():
        if debug:
            print(name)
        data.append({'name':name, 'data':param.data.numpy()})
    
    return data, mv2_cut

def SimpleRunning():

    #### TEST
    ############################ CUSTOM #################################
    arr=[1,2,3,4,5,6,7,8,9]

    i = torch.tensor(arr)
    i = torch.reshape( i, (1,1,3,3))
    w = torch.tensor(arr)
    w= torch.reshape( w, (1,1,3,3))

    conv = torch.nn.Conv2d(in_channels=1,out_channels=3,kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    conv.weight.data = w

    model = conv.eval()## do Inference Mode

    output = model(i)
    print(output)
idx = 0
def customBN(input_data, moving_mean, moving_var, gamma, beta, eps):
    '''
    var = np.sqrt(moving_var + eps)
    factorA = gamma/var
    factorB = beta - factorA*moving_mean
    ## channel wise mult need.
    output = factorA*input_data + factorB
    '''

    ''' do LJM '''
    '''
    
    '''
    var = np.sqrt(moving_var + eps)
    output_data = np.ones(input_data.shape,np.float32)
    factorA = gamma/var
    factorB = beta - factorA*moving_mean

    global idx
    for i in input_data:
        for j in i:
            output_data[0][idx] = factorA[idx]*j + factorB[idx]
            idx = idx + 1
    '''
	print datas JUST FOR TEST!
    '''
    '''
    input_data.astype("float32").tofile(squid_root+"input_data.bin")
    moving_mean.astype("float32").tofile(squid_root+"moving_mean.bin")
    moving_var.astype("float32").tofile(squid_root+"moving_var.bin")
    gamma.astype("float32").tofile(squid_root+"gamma.bin")
    beta.astype("float32").tofile(squid_root+"beta.bin")
    output_data.astype("float32").tofile(squid_root+"output_data.bin")
    factorA.astype("float32").tofile(squid_root+"factorA.bin") 
    var.astype("float32").tofile(squid_root+"var.bin")
    '''
    return output_data

def TESTlayer():
    input_data = np.random.uniform(-1,1,size=(1,3,224,224)).astype('float32')
    test_input_data = torch.tensor(input_data)
    params, base = CutLayer(0,1,True)

    with torch.no_grad():
        valid_data = base[0][0](test_input_data)
        test_input = Numpyize(valid_data)
        valid_data = base[0][1](valid_data)

    valid_data = Numpyize(valid_data)

    BN = base[0][1]
    mean = Numpyize(BN.running_mean)
    var = Numpyize(BN.running_var)
    gamma = params[1]['data']
    beta = params[2]['data']

    output_test = customBN(test_input, mean,var,gamma,beta,1e-05)

    if CMP(output_test,valid_data):
        print("CLEAR")

    else:
        print("NOT")

    return

def GroupConvTest():
    input_data = np.random.uniform(-1,1,size=(1,32,112,112)).astype('float32')
    kernel = np.random.uniform(-1,1,size=(32,8,3,3)).astype('float32')
    i = torch.from_numpy(np.array(input_data))
    k = torch.from_numpy(np.array(kernel))
    conv = torch.nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(3,3), stride=(1,1), padding=(1,1),groups = 4, bias=False)
    conv.weight.data = k

    model = conv.eval()## do Inference Mode
   
    o = model(i)
    output = Numpyize(o)
    input_data.astype("float32").tofile(squid_root+"a.bin")
    kernel.astype("float32").tofile(squid_root+"b.bin")
    output.astype("float32").tofile(squid_root+"c.bin")
    print(output.shape)



    
def BASELINE():


    import torch
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    model.eval()

    from PIL import Image
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))

    # Preprocess the image and convert to tensor
    from torchvision import transforms

    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    img = np.expand_dims(img, 0)

    synset_url = "".join(
        [
            "https://raw.githubusercontent.com/Cadene/",
            "pretrained-models.pytorch/master/data/",
            "imagenet_synsets.txt",
        ]
    )
    synset_name = "imagenet_synsets.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        synsets = f.readlines()

    synsets = [x.strip() for x in synsets]
    splits = [line.split(" ") for line in synsets]
    key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

    class_url = "".join(
        [
            "https://raw.githubusercontent.com/Cadene/",
            "pretrained-models.pytorch/master/data/",
            "imagenet_classes.txt",
        ]
    )
    class_name = "imagenet_classes.txt"
    class_path = download_testdata(class_url, class_name, module="data")
    with open(class_path) as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]

    # Convert input to PyTorch variable and get PyTorch result for comparison
    with torch.no_grad():
        torch_img = torch.from_numpy(img)
        output = model(torch_img)

        # Get top-1 result for PyTorch
        top1_torch = np.argmax(output.numpy())
        torch_class_key = class_id_to_key[top1_torch]

    print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))

    return


#GetParam()
GroupConvTest()
#BASELINE()



