from asyncio import tasks
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
import torch.optim as optim

from dataset.tire_Dataset import TireDataset_Mask,TireDataset,TireDatasetMode
from models.munEfficientformer import efficientformer_l3,efficientformer_l7
from models.EfficientFormer import EfficientFormer
import numpy as np
from torch.nn import functional as F
import os
from utils.dutils import early_stopping

os.environ['CUDA_VISIBLE_DEVICE'] = "0,1"

#configs
root = '/share_dir/Tire/labeled_org/'
batch_size = 16
epochs = 100
device='cuda'
stop_score = 5
cur_score = 0

EfficientFormer_width = {
    'l1': [48, 96, 224, 448],
    'l3': [64, 128, 320, 512],
    'l7': [96, 192, 384, 768],
}
 
EfficientFormer_depth = {
    'l1': [3, 2, 6, 4],
    'l3': [4, 4, 12, 6],
    'l7': [6, 6, 18, 8],
}

model = EfficientFormer(
        layers=EfficientFormer_depth['l7'],
        embed_dims=EfficientFormer_width['l7'],
        downsamples=[True, True, True, True],
        num_classes=5,
        vit_num=4)
# model = model.cuda()
model = nn.DataParallel(model).cuda()


# dataset = TireDatasetMode(root,size=(640,480))
dataset = TireDataset_Mask(root,size=(640,480))
dataset_size = len(dataset)
train_size = int(dataset_size * 0.9)
# testset = TireDataset_Mask(root,mode='test')
trainset, validationset,_ = random_split(dataset, [train_size, dataset_size-train_size,0])
trainloader = DataLoader(trainset,batch_size = batch_size ,shuffle=True,num_workers=0,drop_last=True)
valloader = DataLoader(validationset,batch_size = batch_size ,shuffle=True,num_workers=0,drop_last=True)

# testloader = DataLoader(trainset,batch_size = 1 ,shuffle=True,num_workers=8,drop_last=True)
optimizer = optim.Adam(model.parameters(),lr=0.0001)
weights =[1.0,1.0,1.0,0.7,1.0]
class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='sum')
torch.save(validationset,f'./outputs/classifier/valdiation_data2.dataset')


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
        
        for img, label in tqdm(trainloader,desc='iter',leave=True):
            
            # label = label.unsqueeze(1)
            img = img.cuda()
            out = model(img)
            
            loss = criterion(out,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
            

        avg_loss = np.average(total_loss)
        epoch_bar.set_postfix({'avg_loss': avg_loss})
        print(f"train_loss:{avg_loss}")
        if (epoch+1) % 5 == 0:
            # model.eval()
            with torch.no_grad():
                val_loss = []
            #     for img, label in tqdm(valloader,desc='valdiation'):
                for img, label in tqdm(valloader,desc='val',leave=False):

                    
                    # label = label.unsqueeze(1)
                    img = img.cuda()
                    out = model(img)

                    loss = criterion(out,label)
                    val_loss.append(loss.item())

                avg_val_loss = np.average(val_loss)
                epoch_bar.set_postfix({'avg_val_loss': avg_val_loss})
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(model, f'./outputs/classifier/{epoch}.pt')
                cur_score ,early_stop = early_stopping(best_loss,avg_loss,cur_score,stop_score)
                print(f"train_loss:{avg_loss} validation loss: {avg_val_loss}  best val loss: {best_loss}")
                if early_stop:
                    print('ealry stopping!!!')
                    exit()
            
    #     exit()
