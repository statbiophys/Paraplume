
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0, path=Path("./checkpoint.pt"), best_score=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if score <= self.best_score + self.delta:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})."
            )
            self.val_loss_min = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path.as_posix())

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(1024, 1000)
        self.l2 = nn.Linear(1000,1)
        self.l3 = nn.Linear(1000,1000)

    def forward(self, x):
        x = self.l2F.relu(self.l1(x))
        return torch.sigmoid(x)
