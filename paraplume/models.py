from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paraplume.EGNN import EGNN


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
        self.val_loss_min = np.inf
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

class EGNN_Model(nn.Module):
    '''
    Paragraph uses equivariant graph layers with skip connections
    '''
    def __init__(
        self,
        num_feats,
        edge_dim = 1,
        output_dim = 1,
        graph_hidden_layer_output_dims = None,
        linear_hidden_layer_output_dims = None,
        update_coors = False,
        dropout = 0.0,
        m_dim = 16
    ):
        super(EGNN_Model, self).__init__()

        self.input_dim = num_feats
        self.output_dim = output_dim
        current_dim = num_feats

        # these will store the different layers of out model
        self.graph_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()

        # model with 1 standard EGNN and single dense layer if no architecture provided
        if graph_hidden_layer_output_dims == None: graph_hidden_layer_output_dims = [num_feats]
        if linear_hidden_layer_output_dims == None: linear_hidden_layer_output_dims = []

        # graph layers
        for hdim in graph_hidden_layer_output_dims:
            self.graph_layers.append(EGNN(dim = current_dim,
                                        edge_dim = edge_dim,
                                        update_coors = update_coors,
                                        dropout = dropout,
                                        m_dim = m_dim))
            current_dim = hdim

        # dense layers
        for hdim in linear_hidden_layer_output_dims:
            self.linear_layers.append(nn.Linear(in_features = current_dim,
                                                out_features = hdim))
            current_dim = hdim

        # final layer to get to per-node output
        self.linear_layers.append(nn.Linear(in_features = current_dim, out_features = output_dim))


    def forward(self, feats, coors, edges, mask=None):

        # graph layers
        for layer in self.graph_layers:
            feats = F.hardtanh(layer(feats, coors, edges, mask))

        # dense layers
        for layer in self.linear_layers[:-1]:
            feats = F.hardtanh(layer(feats))

        # output (i.e. prediction)
        feats = self.linear_layers[-1](feats)

        return feats
