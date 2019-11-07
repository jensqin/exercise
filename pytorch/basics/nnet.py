import numpy as np 
import pandas as pd 
import torch
from torch import nn, optim
import torch.nn.functional as F 
import sqlalchemy
from torchvision import datasets, transforms

# import settings
# engine = sqlalchemy.engine_from_config(settings.ENGINE_URL, prefix='FOOTBALL_DEV.')

ftsoff = pd.read_feather('ftsoff.feather')
train = ftsoff[ftsoff['Season'] == 2018]
train_set = train.loc[train['Week'] < 15, 'PassAtt':'TeamRun']
val_set = train.loc[train['Week'] == 15, 'PassAtt':'TeamRun']
test_set = train.loc[train['Week'] == 16, 'PassAtt':'TeamRun']

class simplenet(nn.Module):
    
    def __init__(self):
        super(simplenet, self).__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 20)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

snet = simplenet()
params = list(snet.parameters())
print(len(params))
print(params[0].size())
criterion = nn.MSEloss()
