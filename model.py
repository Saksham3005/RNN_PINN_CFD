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

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4001, 3950)   
        self.fc2 = nn.Linear(3950, 3900)  
        self.fc3 = nn.Linear(3900, 3850)  
        self.fc4 = nn.Linear(3850, 3800)  
        self.fc5 = nn.Linear(3800, 3750)  
        self.fc6 = nn.Linear(3750, 3700)  
        self.fc7 = nn.Linear(3700, 3650)  
        self.fc8 = nn.Linear(3650, 3600)  
        self.fc9 = nn.Linear(3600, 3550)  
        self.fc10 = nn.Linear(3550, 3500)
        self.fc11 = nn.Linear(3500, 3450)
        self.fc12 = nn.Linear(3450, 3400)
        self.fc13 = nn.Linear(3400, 3350)
        self.fc14 = nn.Linear(3350, 3300)
        self.fc15 = nn.Linear(3300, 3250)
        self.fc16 = nn.Linear(3250, 3200)
        self.fc17 = nn.Linear(3200, 3150)
        self.fc18 = nn.Linear(3150, 3100)
        self.fc19 = nn.Linear(3100, 3050)
        self.fc20 = nn.Linear(3050, 3000)


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
        x = torch.tanh(self.fc10(x)) 
        x = torch.tanh(self.fc11(x)) 
        x = torch.tanh(self.fc12(x)) 
        x = torch.tanh(self.fc13(x)) 
        x = torch.tanh(self.fc14(x))
        x = torch.tanh(self.fc15(x)) 
        x = torch.tanh(self.fc16(x)) 
        x = torch.tanh(self.fc17(x)) 
        x = torch.tanh(self.fc18(x)) 
        x = torch.tanh(self.fc19(x)) 
 
        x = self.fc20(x)              
        return x