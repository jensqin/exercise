import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x, y):
        z = self.fc1(x)
        z = torch.cat([z, y], dim=1)
        z = self.fc2(z)
        return z


xx = [torch.rand((2, 3)), torch.rand(2, 2)]
model = MyModel()
model(*xx)

df = pd.DataFrame(1, index=np.arange(10), columns=list("abc"))
y = np.arange(10) + 0.1


class MyData(Dataset):
    def __init__(self, df, y):
        self.df = df
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xx = self.df.loc[idx, :].values
        xx = torch.tensor(xx.tolist(), dtype=torch.float)
        xx2 = torch.rand(2, dtype=torch.float)
        yy = self.y[idx]
        yy = torch.tensor(yy, dtype=torch.float)
        return xx, xx2, yy


example = MyData(df, y)
loader = DataLoader(example, batch_size=2, shuffle=False)

for xx, xx2, yy in loader:
    zz = model(xx, xx2)
    print(zz)
