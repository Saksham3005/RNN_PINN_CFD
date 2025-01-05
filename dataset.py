import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import os
from torchsummary import summary
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DATA(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        df = pd.read_csv(paths)
        
        self.df = df


    def __len__(self):
        return int(len(self.df)/1000) - 1

    def __getitem__(self, idx):
        ui = self.df.iloc[idx*1000 : (1+idx)*1000]['u']
        vi = self.df.iloc[idx*1000 : (1+idx)*1000]['v']
        pi = self.df.iloc[idx*1000 : (1+idx)*1000]['p']
        bi = self.df.iloc[idx*1000 : (1+idx)*1000]['boundary']
        
        x = self.df.iloc[idx*1000 : (1+idx)*1000]['x']
        y = self.df.iloc[idx*1000 : (1+idx)*1000]['y']

        i = idx+1

        uf = self.df.iloc[i*1000 : (1+i)*1000]['u']
        vf = self.df.iloc[i*1000 : (1+i)*1000]['v']
        pf = self.df.iloc[i*1000 : (1+i)*1000]['p']
        bf = self.df.iloc[i*1000 : (1+i)*1000]['boundary']



        sample = {

            'bi': torch.concat([torch.tensor([-999.99],dtype=torch.float32),torch.tensor(bi.to_numpy(), dtype=torch.float32)], dim=0),
            'uf': torch.tensor(uf.to_numpy(), dtype=torch.float32),
            'vf': torch.tensor(vf.to_numpy(), dtype=torch.float32),
            'pf': torch.tensor(pf.to_numpy(), dtype=torch.float32),
            'x' : torch.tensor(x.to_numpy(), dtype=torch.float32),
            'y' : torch.tensor(y.to_numpy(), dtype=torch.float32)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample