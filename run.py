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


#configs
root = 'F:\\data\Tire_data\\220920\\2차데이터_Masked'
root = 'F:\\data\Tire_data\\2210_new'
batch_size = 4
epochs = 100

model = Efficientformer(3,1).cuda()
trainset = TireDataset(root,size=(640,480),mode='train')
# testset = TireDataset_Mask(root,mode='test')
trainloader = DataLoader(trainset,batch_size = batch_size ,shuffle=True,num_workers=0,drop_last=True)

valset = TireDataset(root,size=(640,480),mode='test')
validationloader = DataLoader(valset,batch_size = 2 ,shuffle=True,num_workers=0,drop_last=True)

# testloader = DataLoader(trainset,batch_size = 1 ,shuffle=True,num_workers=8,drop_last=True)
optimizer = optim.Adam(model.parameters(),lr=0.0001)
criterion = nn.MSELoss()
val_criterion = nn.L1Loss()


if __name__== '__main__':

    # for data,label in trainset:
    #     out = model(data)
    #     print(out.shape)
    tqdm(range(epochs)) 

    epoch_bar = tqdm(range(epochs),desc='Epoch ',leave=False)
    best_loss = np.inf

    for epoch in epoch_bar:
        total_loss = []
        model.train()
        for img, label in tqdm(trainloader,desc='iter',leave=False):
            # label = label.unsqueeze(1)
            label = label.cuda()
            img = img.cuda()
            out = model(img)
            loss = criterion(out,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss.append(loss.item())
        avg_loss = np.average(total_loss)
        epoch_bar.set_postfix({'avg_loss': avg_loss})

        if epoch % 5 == 0:
            model.eval()
            val_loss = []
            for img, label in tqdm(validationloader,desc='valdiation'):
                label = label.cuda()
                img = img.cuda()
                out = model(img)
                loss = val_criterion(out,label)
                val_loss.append(loss.item())

            avg_val_loss = np.average(val_loss)
            epoch_bar.set_postfix({'avg_val_loss': avg_val_loss})
            if avg_val_loss < 1.3:
                best_loss = avg_val_loss
                torch.save(model, f'./outputs/Efficientformer/{epoch}.pt')


    #     exit()
