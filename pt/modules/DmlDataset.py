import torch
from torch.utils.data import Dataset

from differential_ml.pt.modules.device import global_device


class DmlDataset(Dataset):
    def __init__(self, x, y, dydx):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.dydx = torch.tensor(dydx, dtype=torch.float32)
        self.x = self.x.to(global_device)
        self.y = self.y.to(global_device)
        self.dydx = self.dydx.to(global_device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {
            'x': self.x[idx],
            'y': self.y[idx],
            'dydx': self.dydx[idx]
        }
        return sample
