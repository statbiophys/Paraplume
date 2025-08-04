"""Implement the dataloader."""

from typing import Any

import torch
import torch.nn.functional as F  # noqa : N812
from torch.utils.data import Dataset


class ParaplumeDataset(Dataset):
    """Create dataset from dictionary of sequences and labels, and from embedidngs."""

    def __init__(
        self,
        dataset_dict: dict[str, dict[str, Any]],
        embeddings: torch.Tensor,
        alphas: list | None = None,
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

    def __getitem__(self, index) -> tuple:
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
        labels_list = get_other_labels(self.dataset_dict, index, alphas=self.alphas)

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

def get_other_labels(
    dataset_dict: dict[str, dict[str, Any]], index: int, alphas: list | None = None
) -> list[torch.Tensor]:
    """Return list of tensors of padded labels for different alphas.

    Args:
        dataset_dict (Dict[str, Dict[str, Any]]): Dictionary mapping indices to positions, \
            imgt numbers and labels.
        index (int): Index of dictionary.
        alphas (Optional[List]): List of alphas to use for multi objective \
            optimization. Defaults to None.

    Returns
    -------
        List[torch.Tensor]: List of padded labels.
    """
    labels_list: list[torch.Tensor] = []
    if alphas is None:
        return labels_list
    for alpha in alphas:
        labels_heavy = dataset_dict[str(index)][f"H_id labels {alpha}"]
        labels_light = dataset_dict[str(index)][f"L_id labels {alpha}"]
        labels_paired = labels_heavy + labels_light
        labels_padded = torch.FloatTensor(
            F.pad(
                torch.FloatTensor(labels_paired),
                (0, 285 - len(torch.FloatTensor(labels_paired))),
                "constant",
                0,
            )
        )
        labels_list.append(labels_padded)
    return labels_list

def create_dataloader(
    dataset_dict: dict[str, dict[str, Any]],
    embeddings: torch.Tensor,
    alphas: list[str] | None = None,
    mode: str = "train",
    batch_size: int = 16,
) -> torch.utils.data.dataloader.DataLoader:
    """Take dataset_dict and embeddings and return dataloader.

    Args:
        dataset_dict (Dict[str, Dict[str, Any]]): Dictionary of sequences and labels.
        embeddings (torch.Tensor): Correspondng embeddings.
        batch_size (int, optional): Batch size. Defaults to 16.
        mode (str): "Returns different dataloader depending on whether its "train", "valid"\
            or "predict" mode.
        alphas (List[str], optional): Alphas to use for multi objective training. Defaults to None.

    Returns
    -------
        torch.utils.data.dataloader.DataLoader: Dataloader to use for training.
    """
    shuffle = mode == "train"
    dataset = ParaplumeDataset(
        dataset_dict=dataset_dict, embeddings=embeddings, alphas=alphas, mode=mode
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
