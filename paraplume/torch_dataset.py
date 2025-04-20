"""Implement the dataloader."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from paraplume.utils_paired import (
    get_other_labels_paired,
)
from paraplume.utils_single import get_other_labels_single


class ParatopeDatasetPairedChain(Dataset):
    """Create dataset from dictionary of sequences and labels, and from embedidngs."""

    def __init__(
        self,
        dataset_dict: Dict[str, Dict[str, Any]],
        embeddings: torch.Tensor,
        alphas: Optional[List] = None,
        mode: str = "train",
    ):
        """Initialize."""
        self.mode = mode
        self.dataset_dict = dataset_dict
        self.embeddings = embeddings
        self.alphas = alphas

    def __len__(self):
        """Return number of sequences in dataset."""
        return self.embeddings.shape[0]

    def __getitem__(self, index) -> Tuple:
        """Return embeddings and labels for training model for given index."""
        main_labels_heavy = self.dataset_dict[str(index)]["H_id labels 4.5"]
        main_labels_light = self.dataset_dict[str(index)]["L_id labels 4.5"]
        numbers_heavy = self.dataset_dict[str(index)]["H_id numbers"]
        numbers_light = self.dataset_dict[str(index)]["L_id numbers"]
        pdb_code = self.dataset_dict[str(index)]["pdb_code"]
        embedding = self.embeddings[index, :, :]
        main_labels_paired = main_labels_heavy + main_labels_light
        main_labels = torch.FloatTensor(
            F.pad(
                torch.FloatTensor(main_labels_paired),
                (0, 285 - len(torch.FloatTensor(main_labels_paired))),
                "constant",
                0,
            )
        )
        len_heavy = len(main_labels_heavy)
        len_light = len(main_labels_light)
        labels_list = get_other_labels_paired(self.dataset_dict, index, alphas=self.alphas)

        if self.mode == "train":
            return (
                embedding,
                main_labels,
                len_heavy,
                len_light,
                *labels_list,
            )
        if self.mode == "test":
            return (embedding, main_labels, len_heavy, len_light)
        if self.mode == "predict":
            return (
                embedding,
                main_labels,
                len_heavy,
                len_light,
                pdb_code,
                numbers_heavy,
                numbers_light,
            )
        raise ValueError("Invalid mode. Choose from 'train', 'test', 'predict'")

class ParatopeDatasetSingleChain(Dataset):
    """Create dataset from dictionary of sequences and labels, and from embedidngs."""

    def __init__(
        self,
        dataset_dict: Dict[str, Dict[str, Any]],
        embeddings: torch.Tensor,
        alphas: Optional[List] = None,
        mode: str = "train",
    ):
        """Initialize."""
        self.mode = mode
        self.dataset_dict = dataset_dict
        self.embeddings = embeddings
        self.alphas = alphas

    def __len__(self):
        """Return number of sequences in dataset."""
        return self.embeddings.shape[0]

    def __getitem__(self, index) -> Tuple:
        """Return embeddings and labels for training model for given index."""
        main_labels = self.dataset_dict[str(index)]["labels 4.5"]
        length = len(main_labels)
        numbers = self.dataset_dict[str(index)]["numbers"]
        pdb_code = self.dataset_dict[str(index)]["pdb_code"]
        embedding = self.embeddings[index, :, :]
        main_labels = torch.FloatTensor(
            F.pad(
                torch.FloatTensor(main_labels),
                (0, 285 - len(torch.FloatTensor(main_labels))),
                "constant",
                0,
            )
        )

        labels_list = get_other_labels_single(self.dataset_dict, index, alphas=self.alphas)

        if self.mode == "train":
            return (
                embedding,
                main_labels,
                length,
                *labels_list,
            )
        if self.mode == "test":
            return (embedding, main_labels, length)
        if self.mode == "predict":
            return (
                embedding,
                main_labels,
                length,
                pdb_code,
                numbers,
            )
        raise ValueError("Invalid mode. Choose from 'train', 'test', 'predict'")

def create_dataloader(
    dataset_dict: Dict[str, Dict[str, Any]],
    embeddings: torch.Tensor,
    alphas: Optional[List[str]] = None,
    mode: str = "train",
    batch_size: int = 16,
    chain="paired",
) -> torch.utils.data.dataloader.DataLoader:
    """Take dataset_dict and embeddings and return dataloader.

    Args:
        dataset_dict (Dict[str, Dict[str, Any]]): Dictionary of sequences and labels.
        embeddings (torch.Tensor): Correspondng embeddings.
        batch_size (int, optional): Batch size. Defaults to 16.
        mode (str): "Returns different dataloader depending on whether its "train", "valid"\
            or "predict" mode.
        alphas (List[str], optional): Alphas to use for multi objective training. Defaults to None.

    Returns:
        torch.utils.data.dataloader.DataLoader: Dataloader to use for training.
    """
    if chain=="paired":
        shuffle = mode == "train"
        dataset = ParatopeDatasetPairedChain(
            dataset_dict=dataset_dict, embeddings=embeddings, alphas=alphas, mode=mode
        )
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset_loader
    shuffle = mode == "train"
    dataset = ParatopeDatasetSingleChain(
        dataset_dict=dataset_dict, embeddings=embeddings, alphas=alphas, mode=mode
    )
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader
