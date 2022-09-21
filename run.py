import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.tire_Dataset import TireDataset_Mask
from models.munEfficientformer import efficientformer_l3,efficientformer_l7


#configs
root = 'F:\\data\Tire_data\\220920\\2차데이터_Masked'
batch_size = 4


model = efficientformer_l3()
trainset = TireDataset_Mask(root,size=(448,448),mode='train')
# testset = TireDataset_Mask(root,mode='test')
trainloader = DataLoader(trainset,batch_size = batch_size ,shuffle=True,num_workers=0,drop_last=True)
# testloader = DataLoader(trainset,batch_size = 1 ,shuffle=True,num_workers=8,drop_last=True)



if __name__== '__main__':

    # for data,label in trainset:
    #     out = model(data)
    #     print(out.shape)
        

    for img, label in trainloader:
        print(img.shape)
        print(label.shape)
        out = model(img)
        exit()

    #     exit()
