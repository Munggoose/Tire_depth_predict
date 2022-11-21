from pickletools import float8
from PIL import Image
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
from utils import split_img
import json
import cv2


#tire Dataset class
class TireDatasetSplit(Dataset):
    def __init__(self,root_path, excel_path,custom_transforms=None):
        """_summary_

        Args:
            root_path (str): data root folder
            excel_path (str): tire_data label excel-> 'tire_result.xlsx' path
            custom_transforms (torchvision.transformers, optional): custom transforms method 

        """
        self.split_cnt = 3
        if custom_transforms:
            self.transforms = custom_transforms

        else:
            #org:3024 x 4032 default transforms
            # self.transforms = transforms.Compose([transforms.Resize((512,672)),
            #                             transforms.ToTensor()])
            self.transforms = transforms.Compose([transforms.Resize((480,640)),
                            transforms.ToTensor()])
            
            # self.transforms = transforms.Compose([transforms.Resize((252,336)),
            #                             transforms.ToTensor(),
            #                             transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])]
            # /self.label_transforms = transforms.Compose([transforms.ToTensor()])
        # label = torch.FloatTensor(label)
        
        self.img_paths = [] 
        self.label = []

        with open(excel_path,'rb') as f:
            label = pd.read_excel(f,)
        depth = label[['depth1', 'depth2', 'depth3','depth4', 'depth5', 'depth6', 'depth7', 'depth8', 'depth9', 'depth10','depth11', 'depth12']].apply(np.average,axis=1)
        label['depth'] = depth
        label['class'] = label['depth'].apply(round) #get Label -> average depth from depth points 

        label = label[['sid','class']] #sid:folder name ,class: label(average depth)


        for folder_name, cls in label[['sid','class']].values:
            f_path = os.path.join(root_path,str(folder_name))

            data_paths = glob(f'{f_path}/*.jpg')
            for img in data_paths:
                self.img_paths.append(img)
                self.label.append(cls)
    
    def __len__(self):
        return len(self.img_paths*(self.split_cnt**2))


    def __getitem__(self, idx):
        """_summary_

        Returns:
            dict{'image': tensor, 'label': int }
        """
        
        sub_idx = idx %(self.split_cnt**2)
        idx = idx//(self.split_cnt**2)
        label = self.label[idx]
        image = Image.open(self.img_paths[idx])
        image = np.array(image)
        images = split_img(image)
        # image = np.array(image)
        sample = {"image":images[sub_idx],'label':torch.tensor(label,dtype=float)}
        sample['image'] = self.transforms(sample['image'])
        print(label)
        return sample


class TireDataset(Dataset):
    def __init__(self,root_path,size=(640,480), excel_path='F:\\data\Tire_data\\2210_new\\tire_result.xlsx',custom_transforms=None,mode ='train'):
        """_summary_

        Args:
            root_path (str): data root folder
            excel_path (str): tire_data label excel-> 'tire_result.xlsx' path
            custom_transforms (torchvision.transformers, optional): custom transforms method 

        """

        root_path = os.path.join(root_path,mode)
        self.split_cnt = 3
        if custom_transforms:
            self.transforms = custom_transforms


        else:
            #org:3024 x 4032 default transforms
            # self.transforms = transforms.Compose([transforms.Resize((640,480)),
            #                 transforms.ToTensor(),
            #                 transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
            self.transforms = transforms.Compose([transforms.Resize(size),
                            transforms.ToTensor(),
                            ])
            
            # self.transforms = transforms.Compose([transforms.Resize((252,336)),
            #                             transforms.ToTensor(),
            #                             transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])]
            # /self.label_transforms = transforms.Compose([transforms.ToTensor()])
        # label = torch.FloatTensor(label)
        self.img_paths = [] 
        self.label = []
        
        with open(excel_path,'rb') as f:
            label = pd.read_excel(f,)
        depth = label[['depth1', 'depth2', 'depth3','depth4', 'depth5', 'depth6', 'depth7', 'depth8', 'depth9', 'depth10','depth11', 'depth12']].apply(np.average,axis=1)
        label['depth'] = depth

        # label['class'] = label['depth'].apply(round ) #get Label -> average depth from depth points 
        result  = label[['depth']].apply(lambda x: round(x,2) )

        label['class'] = result

        label = label[['sid','class']] #sid:folder name ,class: label(average depth)


        for folder_name, cls in label[['sid','class']].values:
            
            f_path = os.path.join(root_path,str(int(folder_name)))
            data_paths = glob(f'{f_path}/*.jpg')

            for img in data_paths:
                self.img_paths.append(img)
                self.label.append(cls)

    
    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        """_summary_


        Returns:
            dict{'image': tensor, 'label': int }
        """

        label = self.label[idx]
        image = Image.open(self.img_paths[idx])
        # image = np.array(image)
        # image = np.array(image) #dtype=torch.float32, device=self.device
        sample = {"image":image,'label':torch.tensor(label,dtype=torch.float32)}
        sample['image'] = self.transforms(sample['image'])
        return sample['image'], sample['label']


class TireDataset_Mask(Dataset):

    def __init__(self, root,size= (640,480), custom_transform=None,device='cuda'):
        self.device = device
        self.images = glob(os.path.join(root,'**/**/*.jpg'))
        self.mask_imgs = []
        self.label_dict = {'1.5': 0,'4.0': 1,'4.5':2,'5.5':3,'7.0':4}
        self.make_label()
        
        if custom_transform:
            self.transform = custom_transform
        else:
            self.transforms = transforms.Compose([transforms.Resize(size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),])
            
    def make_label(self):
        labels = []
        for img_path in self.images:
            label = self.label_dict[img_path.split('/')[-3]]
            img = self.mask_img(img_path)
            # label = torch.tensor(label, device=self.device, dtype=torch.int8)
            labels.append(label)
            self.mask_imgs.append(img)
        self.labels = torch.tensor(labels,device=self.device)


        
    def mask_img(self, img_path):
        json_path = img_path[:-3] + 'json'
        img = Image.open(img_path)
        with open(json_path, 'r') as f:
            j_data = json.load(f)
        points = j_data['shapes'][0]['points']
        mask1 = np.zeros((img.size),dtype = np.uint8)
        mask_img = cv2.fillPoly(mask1,np.int32([points]),1)
        img_arr = np.array(img)
        mask_img = np.expand_dims(mask_img, axis=2)
        masked_img = np.transpose(img_arr,[1,0,2]) * mask_img
        img = Image.fromarray(masked_img)
        
        return img
    
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self,idx):
        img = self.transforms(self.mask_imgs[idx])
        label = self.labels[idx]
        return (img, label)
    



class TireDataset_Mask2(Dataset):

    def __init__(self, root,size= (640,480) ,mode='train', custom_transform=None,device='cuda',csv_root='./dataset/tire_result.xlsx'):

        self.device = device
        self.mode = mode
        self.img_list = None
        self.root = root
        self.load_label_csv(csv_root)
        if custom_transform:
            self.transforms = custom_transform

        else:
            self.transforms = transforms.Compose([transforms.Resize(size),
                            transforms.ToTensor()])

        self.load_image(root)
    

    def load_label_csv(self,csv_root):
        self.label = []
        
        with open(csv_root,'rb') as f:
            label = pd.read_excel(f)
        depth = label[['depth1', 'depth2', 'depth3','depth4', 'depth5', 'depth6', 'depth7', 'depth8', 'depth9', 'depth10','depth11', 'depth12']].apply(np.average,axis=1)

        label['depth'] = depth

        # label['class'] = label['depth'].apply(round ) #get Label -> average depth from depth points 
        result  = label[['depth']].apply(lambda x: round(x,2) )

        label['class'] = result

        label = label[['sid','class']] #sid:folder name ,class: label(average depth)
        self.label = {}
        for folder_name, cls in label[['sid','class']].values:
            self.label[str(int(folder_name))] = cls

        # for folder_name, cls in label[['sid','class']].values:
            
        #     f_path = os.path.join(self.root,str(int(folder_name)))
        #     data_paths = glob(f'{f_path}/*.jpg')

        #     for img in data_paths:
        #         self.img_paths.append(img)
        #         self.label.append(cls)


    def load_image(self,root):
        # print(os.path.join(root,'**/*.jpg'))
        self.img_list = glob(os.path.join(root,'**/*.png'),recursive=True)
        # print(len(self.img_list))


        # self.images = [img_loader(img_path) for img_path in img_list]
        # self.labels = [label_maker(img_path) for img_path in img_list]

    def label_maker(self,path):
        label = os.path.dirname(path).split('/')[-1].split('\\')[-1]

        label = self.label[label]
        return torch.tensor(float(label), dtype=torch.float32, device=self.device)


    def img_loader(self,path):
        img = Image.open(path)
        return self.transforms(img)


    def __len__(self):
        return len(self.img_list)
    

    def __getitem__(self, idx):
        
        path = self.img_list[idx]
        img = self.img_loader(path)
        label = self.label_maker(path)
        return img, label


class TireDatasetMode(Dataset):
    
    def __init__(self, root,size= (640,480), custom_transform=None,device='cuda'):
        self.device = device
        self.images = glob(os.path.join(root,'**/**/*.jpg')) 
        self.label_dict = {'1.5': 0,'4.0': 1,'4.5':2,'5.5':3,'7.0':4}
        self.make_label()
        
        if custom_transform:
            self.transform = custom_transform
        else:
            self.transforms = transforms.Compose([transforms.Resize(size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),])
            
    def make_label(self):
        labels = []
        for img_path in self.images:
            label = self.label_dict[img_path.split('/')[-3]]
            # label = torch.tensor(label, device=self.device, dtype=torch.int8)
            labels.append(label)
        self.labels = torch.tensor(labels,device=self.device)
    
    def load_image(self, x):
        img = Image.open(x)
        return self.transforms(img)
    
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self,idx):
        img = self.load_image(self.images[idx])
        label = self.labels[idx]
        return (img, label)