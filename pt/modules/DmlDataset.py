import torch
from torch.utils.data import Dataset


class DmlDataset(Dataset):
    def __init__(self, x, y, dydx):
        self.x = x
        self.y = y
        self.dydx = dydx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {
            'x': torch.as_tensor(self.x[idx], dtype=torch.float32),
            'y': torch.as_tensor(self.y[idx], dtype=torch.float32),
            'dydx': torch.as_tensor(self.dydx[idx], dtype=torch.float32)
        }
        return sample
