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

nu = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
weights_folder = "params"

path_joined = []
for j in range(0, 25):
    path_joined.append(f"use/data_new{j}.csv")
    

l = []
for i in range(0, 100):
    l.append(i)
l = np.array(l)

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4001, 3000)   
        self.fc2 = nn.Linear(3000, 2000)  
        self.fc3 = nn.Linear(2000, 1000)  
        self.fc4 = nn.Linear(1000, 500)  
        self.fc5 = nn.Linear(500, 100)  
        self.fc6 = nn.Linear(100, 500)  
        self.fc7 = nn.Linear(500, 1000)  
        self.fc8 = nn.Linear(1000, 2000)  
        self.fc9 = nn.Linear(2000, 2500)  
        self.fc10 = nn.Linear(2500, 3000)
        # self.fc11 = nn.Linear(350, 345)
        # self.fc12 = nn.Linear(345, 340)
        # self.fc13 = nn.Linear(340, 335)
        # self.fc14 = nn.Linear(335, 330)
        # self.fc15 = nn.Linear(330, 325)
        # self.fc16 = nn.Linear(325, 320)
        # self.fc17 = nn.Linear(320, 315)
        # self.fc18 = nn.Linear(315, 310)
        # self.fc19 = nn.Linear(310, 305)
        # self.fc20 = nn.Linear(305, 3000)


    def forward(self, x):
        x = torch.tanh(self.fc1(x))  
        x = torch.tanh(self.fc2(x))  
        x = torch.tanh(self.fc3(x))  
        x = torch.tanh(self.fc4(x)) 
        x = torch.tanh(self.fc5(x)) 
        x = torch.tanh(self.fc6(x)) 
        x = torch.tanh(self.fc7(x)) 
        x = torch.tanh(self.fc8(x)) 
        x = torch.tanh(self.fc9(x))
        # x = torch.tanh(self.fc10(x)) 
        # x = torch.tanh(self.fc11(x)) 
        # x = torch.tanh(self.fc12(x)) 
        # x = torch.tanh(self.fc13(x)) 
        # x = torch.tanh(self.fc14(x))
        # x = torch.tanh(self.fc15(x)) 
        # x = torch.tanh(self.fc16(x)) 
        # x = torch.tanh(self.fc17(x)) 
        # x = torch.tanh(self.fc18(x)) 
        # x = torch.tanh(self.fc19(x)) 
 
        x = self.fc10(x)              
        return x

class DATA(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        df = pd.read_csv(paths)
        
        self.df = df


    def __len__(self):
        return int(len(self.df)/1000)

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
      


model = Net().to(device)
model.load_state_dict(torch.load('params/PINN_attempt_1_81.pt'))
model.eval()
optimizer = optim.Adam(model.parameters(), lr=0.001)
k = 0

for time, pt in enumerate(path_joined):
    
    PINN_dataset = DATA(pt)
    data = DataLoader(PINN_dataset, batch_size=1, pin_memory=True, num_workers=2)
    t = torch.ones((1, 1000), dtype=torch.float32).to(device) * (time + 1)

    with torch.no_grad():
            
        for i, sample in enumerate(data):
            
            x = sample['x'].float().to(device)
            print(x.shape)
            y = sample['y'].float().to(device)
            b = sample['bi'].float().to(device)
            
            bi = sample['bi']
            
            input = torch.hstack((x, y, t, b)).to(device)
            input = torch.squeeze(input, dim=0).to(device)
            output = model(input)

            
            if output.shape[0] != 3000:
                u = output[:, torch.arange(0, 3000, 3)].cpu().numpy()
                v = output[:, torch.arange(1, 3000, 3)].cpu().numpy()
                p = output[:, torch.arange(2, 3000, 3)].cpu().numpy()
            else:
                v = output[torch.arange(1, 3000, 3)].cpu().numpy()
                p = output[torch.arange(2, 3000, 3)].cpu().numpy()
                u = output[torch.arange(0, 3000, 3)].cpu().numpy()


            u_pred = u 
            v_pred = v 
            p_pred = p 
            
            # Save predictions
            predictions = pd.DataFrame({
                'u_pred': u_pred,
                'v_pred': v_pred,
                'p_pred': p_pred.flatten(),

            })
            predictions.to_csv(f'output_new/predictions_{k}_{i}.csv', index=False)
    k += 1
print("Predictions saved to individual CSV files")
