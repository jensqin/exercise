from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

X, y = load_boston(return_X_y=True)
X_ts = torch.as_tensor(X, dtype=torch.float)
y_ts = torch.as_tensor(y, dtype=torch.float)

class BostonData(Dataset):
    """
    boston dataset
    """
    pass

kf = KFold(10)
result = list(kf.split(X))
