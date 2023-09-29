import torch

from differential_ml.pt.modules.device import global_device


class SimpleDataLoader:
    def __init__(self, x, y, dy_dx, batch_size=1, shuffle=True):
        self.x = torch.tensor(x, dtype=torch.float32, device=global_device)
        self.y = torch.tensor(y, dtype=torch.float32, device=global_device)
        self.dy_dx = torch.tensor(dy_dx, dtype=torch.float32, device=global_device)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = list(range(len(self.x)))
        self._shuffle()

    def _shuffle(self):
        if self.shuffle:
            self.indexes = torch.randperm(len(self.indexes))

    def __iter__(self):
        self._shuffle()

        for i in range(0, len(self.indexes), self.batch_size):
            batch_indexes = self.indexes[i:i + self.batch_size]
            # batch_data = [self.dataset[idx] for idx in batch_indexes]
            # batch = [torch.stack(samples) for samples in zip(*batch_data)]
            yield {
                'x': self.x[self.indexes],
                'y': self.y[self.indexes],
                'dydx': self.dy_dx[self.indexes]
            }


    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size
