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
from sklearn.metrics import classification_report


os.environ['CUDA_VISIBLE_DEVICE'] = "0,1"

#configs
root = '/share_dir/Tire/test/'
batch_size = 2
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

# model = EfficientFormer(
#         layers=EfficientFormer_depth['l3'],
#         embed_dims=EfficientFormer_width['l3'],
#         downsamples=[True, True, True, True],
#         num_classes=5,
#         vit_num=4)

model = torch.load('./outputs/classifier/29.pt')
model = model.cuda()


dataset = torch.load('./outputs/classifier/valdiation_data2.dataset') 
# dataset = TireDatasetMode(root,size=(640,480))

testloader = DataLoader(dataset,batch_size = batch_size ,shuffle=True,num_workers=0,drop_last=True)

if __name__== '__main__':
    total_pred = []
    total_gt = []
    with torch.no_grad():
        for img, label in tqdm(testloader,desc='test',leave=True):

            img = img.cuda()
            
            out = model(img)
            pred = torch.argmax(out,axis=1)
            pred = pred.detach().to('cpu').numpy()
            label = label.detach().to('cpu').numpy()
            # print(f"pred {pred} : gt : {label}")
            total_pred.extend(pred)
            total_gt.extend(label)
    print(total_pred)
    print(total_gt)
    target_names = ['1.5', '4.0', '4.5','5.5','7.0']
    print(classification_report(total_gt, total_pred, target_names=target_names))
