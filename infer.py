from asyncio import tasks
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

from dataset.tire_Dataset import TireDataset_Mask,TireDataset
from models.munEfficientformer import efficientformer_l3,efficientformer_l7
from models.Efficient import Efficientformer
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
from glob import glob 


#configs
root = 'F:\\data\Tire_data\\2210_new'
batch_size = 1

# model = Efficientformer(3,1).cuda()

model = torch.load('./outputs\Efficientformer/90.pt')
transforms_f = transforms.Compose([transforms.Resize((640,480)),
                            transforms.ToTensor(),
                            ])



if __name__== '__main__':


    model.eval()
    img_list = glob('./data/**/*.jpg')

    for img_path in tqdm(img_list):
        label = os.path.basename(os.path.dirname(img_path))
        name = os.path.basename(img_path)
 
        img = Image.open(img_path)
        img = transforms_f(img).unsqueeze(0).cuda()

        out = model(img).squeeze(0)
        print(f'{name}_ {label} __ {out.item()}')



    #     exit()
