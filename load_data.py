import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import os

import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
import cv2

import albumentations as albu
from albumentations.pytorch import ToTensorV2
import random
random.seed(9000)

from pycox.models import LogisticHazard
import torchtuples as tt

    
class CPMPdataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_path, transform=None, mode='train'):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        df = pd.read_csv(self.csv_path, encoding='cp949', converters={'filename':str})
        
        if self.mode == 'test':
            idx_lst = list(df[df.split == 'test'].filename)
        elif self.mode == 'val':
            idx_lst = list(df[df.split == 'val'].filename)
        elif self.mode == 'train':
            idx_lst = list(df[df.split == 'train'].filename)
        elif self.mode == 'ext':
            idx_lst = list(df[df.split == 'ext'].filename)
        elif self.mode == 'infer':
            idx_lst = [os.path.splitext(n)[0] for n in sorted(os.listdir(self.data_dir))]
        else:
            print("Mode error!")
            

        path_lst = [os.path.join(self.data_dir, x+'.png') for x in idx_lst]        
        time_lst = [df[df.filename == x].fu_dur.values[0] for x in idx_lst]
        event_lst = [df[df.filename == x].death.values[0] for x in idx_lst]
        target = (np.array(time_lst), np.array(event_lst))
        
        labtrans = LogisticHazard.label_transform(20)
        time_event = labtrans.fit_transform(*target)

        self.time, self.event = tt.tuplefy(time_event[0], time_event[1]).to_tensor()

        self.imgs_lst = path_lst
        self.labtrans = labtrans

    def __len__(self):
        return len(self.imgs_lst)

    def __getitem__(self, index):
    
        img = np.array(Image.open(self.imgs_lst[index]).convert("RGB"))
        
        if self.transform:
            img = self.transform(image=img)
        
        data = img['image'], (self.time[index], self.event[index])

        return data
    

class CPMPinferdataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        idx_lst = [os.path.splitext(n)[0] for n in sorted(os.listdir(self.data_dir))]
        path_lst = [os.path.join(self.data_dir, x+'.png') for x in idx_lst]       
        self.imgs_lst = path_lst
        
    def __len__(self):
        return len(self.imgs_lst)

    def __getitem__(self, index):
    
        img = np.array(Image.open(self.imgs_lst[index]).convert("RGB"))
        
        if self.transform:
            img = self.transform(image=img)

        return img['image']
    

def load_data(batchsize, data_dir, csv_path):

    print('batchsize: ', batchsize)
    

    data_transforms = {
    'train': albu.Compose([
         albu.Resize(256, 256),
        albu.OneOf([
            albu.RandomBrightness(limit=.2, p=1), 
            albu.RandomContrast(limit=.2, p=1), 
            albu.RandomGamma(p=1)
        ], p=.3),
        albu.OneOf([
            albu.Blur(blur_limit=3, p=1),
            albu.MedianBlur(blur_limit=3, p=1)
        ], p=.2),
        albu.OneOf([
            albu.GaussNoise(0.002, p=.5),
        ], p=.2),
         albu.RandomRotate90(p=.5),
         albu.HorizontalFlip(p=.5),
         albu.VerticalFlip(p=.5),
#         albu.Cutout(num_holes=10, 
#                     max_h_size=int(.1 * size), max_w_size=int(.1 * size), 
#                     p=.25),
        albu.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.01, rotate_limit=10, p=0.4, border_mode = cv2.BORDER_CONSTANT),
        albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'val': albu.Compose([
         albu.Resize(256, 256),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'test': albu.Compose([
         albu.Resize(256, 256),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'ext': albu.Compose([
         albu.Resize(256, 256),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    }

    image_datasets = {x: CPMPdataset(data_dir, csv_path, mode=x, transform=data_transforms[x])
              for x in ['train', 'val', 'test', 'ext']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                         shuffle=True, num_workers=8)
          for x in ['train', 'val', 'test', 'ext']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test', 'ext']}
    labtrans = CPMPdataset(data_dir, csv_path, mode='train', transform=data_transforms['train']).labtrans

    return image_datasets, dataloaders, dataset_sizes, labtrans


def load_infer_data(batchsize, data_dir, csv_path, train_data_dir, train_csv_path):
    
    print('batchsize: ', batchsize)
    
    data_transforms = albu.Compose([
         albu.Resize(256, 256),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ])
    
    image_datasets = CPMPinferdataset(data_dir, transform=data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batchsize,
                                         shuffle=True, num_workers=8)
    dataset_sizes = len(image_datasets)
    labtrans = CPMPdataset(train_data_dir, train_csv_path, mode='train').labtrans


    return image_datasets, dataloaders, dataset_sizes, labtrans
