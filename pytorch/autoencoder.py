import pandas as pd
import seaborn as sns
import torch
from torch import nn
import torch.nn.functional as F


ftsoff = pd.read_feather('ftsoff.feather')

train = ftsoff.loc[:, 'Season':'TeamRun']

class autoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(31, 20)
        self.fc2 = nn.Linear(20, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
        
class denoise(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(31, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, 31)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

