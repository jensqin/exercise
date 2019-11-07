import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import sqlalchemy
# import settings

# engine = sqlalchemy.engine_from_config(settings.ENGINE_URL, prefix='FOOTBALL_DEV.')


x = torch.rand(3, 4)

class net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)
        self.fc3 = nn.Linear(1, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x)
        return x
