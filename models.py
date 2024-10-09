
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim_x, input_dim_y):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_dim_x * input_dim_y, 1000)
        self.l2 = nn.Linear(1000, input_dim_x)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l2(F.relu(self.l1(x)))
        return torch.sigmoid(x)

class CNN(nn.Module):
    def __init__(self, input_dim_x, input_dim_y, ):
        super(CNN, self).__init__()

        # First Convolutional Layer (with batch normalization)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Pooling layer (reduces spatial dimension)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * (input_dim_x // 4) * (input_dim_y // 4), 1000)  # Adjust for pooling
        self.fc2 = nn.Linear(1000, input_dim_x)

    def forward(self, x):
        # Apply first conv layer, batch norm, ReLU, and pooling
        a=self.conv1(x)
        x = self.pool(F.relu(self.bn1(a)))

        # Apply second conv layer, batch norm, ReLU, and pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Flatten the output of conv layers before passing to FC
        x = x.view(x.shape[0], -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Apply sigmoid activation for binary classification
        return torch.sigmoid(x)

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
