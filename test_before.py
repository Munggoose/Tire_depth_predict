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


#configs
root = 'F:\\data\Tire_data\\2210_new'
batch_size = 1

# model = Efficientformer(3,1).cuda()

valset = TireDataset(root,size=(640,480),mode='test')
validationloader = DataLoader(valset,batch_size = batch_size ,shuffle=True,num_workers=0,drop_last=True)
model = torch.load('./outputs\Efficientformer/80.pt')
# testloader = DataLoader(trainset,batch_size = 1 ,shuffle=True,num_workers=8,drop_last=True)
optimizer = optim.Adam(model.parameters(),lr=0.0001)
criterion = nn.MSELoss()
val_criterion = nn.L1Loss()


if __name__== '__main__':

    best_loss = np.inf
    max_diff = -np.inf
    model.eval()
    val_loss = []
    labels = []
    preds = []
    for img, label in tqdm(validationloader,desc='valdiation'):
        label = label.cuda()
        img = img.cuda()
        out = model(img).squeeze(0)
        
        loss = val_criterion(out,label)
        diff = loss.item()
        val_loss.append(diff)
        
        if diff > max_diff:
            max_diff = diff
        preds.append(out.cpu().detach().numpy()[0] )
        labels.append(label.cpu().detach().numpy()[0] )



    avg_val_loss = np.average(val_loss)
    print(avg_val_loss)
    print(max_diff)
    result ={'labels': labels,'pred':preds}
    df = pd.DataFrame(result)
    df.to_csv('result.csv')


    #     exit()
