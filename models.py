
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
    def __init__(self, dropout_prob=0, dim1 = 1000, dim2=1, dim3=1, batch_norm=False):
        super(MLP, self).__init__()
        self.batch_norm=batch_norm
        self.one_hidden=dim2==1
        self.two_hidden = (dim2>1 and dim3==1)
        self.three_hidden = dim3>1
        self.dropout = nn.Dropout(p=dropout_prob)
        self.l1 = nn.Linear(2048, dim1)
        self.l2 = nn.Linear(dim1, dim2)
        self.l3 = nn.Linear(dim2,dim3)
        self.l4 = nn.Linear(dim3,1)
        self.bn1 = nn.BatchNorm1d(dim1)  # Batch normalization for first layer
        self.bn2 = nn.BatchNorm1d(dim2)
        self.bn3 = nn.BatchNorm1d(dim3)

    def forward(self, x):
        x1=self.l1(x)
        if self.batch_norm:
            x1=self.bn1(x1)
        x2=self.l2(self.dropout(F.relu(x1)))
        if self.one_hidden :
            x=x2
            return torch.sigmoid(x)
        if self.batch_norm:
            x2=self.bn2(x2)
        x3 = self.l3(self.dropout(F.relu(x2)))
        if self.two_hidden:
            x=x3
            return torch.sigmoid(x)
        if self.batch_norm:
            x3=self.bn3(x3)
        x4 = self.l4(self.dropout(F.relu(x3)))
        return torch.sigmoid(x4)

class MLP_final(nn.Module):
    def __init__(self, dim = 3):
        super(MLP_final, self).__init__()
        self.l1 = nn.Linear(dim,1)
        with torch.no_grad():
            print(self.l1.weight.shape)# Disable gradient updates during initialization
            self.l1.weight.fill_(0)  # Set all weights to 0 initially
            self.l1.weight[:,0] = 1  # Set the first weight to 1
            self.l1.bias.fill_(0)
        self.l2 = nn.Linear(3, 1)

    def forward(self, x):
        #return torch.sigmoid(self.l2(F.relu(self.l1(x))))
        return torch.sigmoid(self.l1(x))
