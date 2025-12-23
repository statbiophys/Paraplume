"""Implement the dataloader."""

from typing import Any

import torch
from torch.utils.data import Dataset


class ParaplumeDataset(Dataset):
    """
    Dataset for antibody-antigen residue-residue interaction prediction.

    Each sample corresponds to a pair (i, j):
        - i = residue index in antibody
        - j = residue index in antigen
        - label = 1 if residues i and j are in contact (≤ 4.5 Å), else 0

    Args:
        dataset_dict (dict): Dictionary built by build_dictionary().
        antibody_embeddings (torch.Tensor): Antibody per-residue embeddings, shape (N, L_ab, D_ab).
        mode (str): Whether to load the dataset in training or testing mode.
    """

    def __init__(
        self,
        dataset_dict: dict[str, dict[str, Any]],
        antibody_embeddings: torch.Tensor,
        mode="train",
    ):
        self.antibody_embeddings = antibody_embeddings
        # Build an index mapping (sample_idx, i, j) for all pairs
        self.dataset = []
        self.mode = mode
        for sample_idx, value in dataset_dict.items():
            sample_idx_int = int(sample_idx)

            len_heavy = len(value["H_id labels 4.5"])
            pdb_code = value["pdb_code"]
            labels_ab = value["H_id labels 4.5"] + value["L_id labels 4.5"]
            numbers_ab = value["H_id numbers"] + value["L_id numbers"]
            sequence = value["H_id sequence"] + value["L_id sequence"]

            for i in range(len(labels_ab)):
                label = labels_ab[i]
                chain = "H"
                real_pos = i
                if i >= len_heavy:
                    chain = "L"
                    real_pos = i - len_heavy
                num = numbers_ab[i]
                aa = sequence[i]
                self.dataset.append((sample_idx_int, label, num, i, real_pos, chain, pdb_code, aa))

    def __len__(self):
        """Return total number of antibody-antigen residue pairs."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return embedding pair and label for (antibody residue i, antigen residue j)."""
        (sample_idx, label, num, i, real_pos, chain, pdb_code, aa) = self.dataset[idx]
        # Retrieve embeddings
        ab_emb = self.antibody_embeddings[sample_idx][i]  # shape (D_ab,)

        if self.mode == "test":
            return (sample_idx, ab_emb, label, num, i, real_pos, chain, pdb_code, aa)
        return (
            sample_idx,
            ab_emb,
            label,
            num,
            i,
            real_pos,
        )


def create_dataloader(
    dataset_dict: dict[str, dict[str, Any]],
    embeddings: torch.Tensor,
    batch_size: int = 16,
    mode: str = "train",
) -> torch.utils.data.dataloader.DataLoader:
    """Take dataset_dict and embeddings and return dataloader.

    Args:
        dataset_dict (Dict[str, Dict[str, Any]]): Dictionary of sequences and labels.
        embeddings (torch.Tensor): Corresponding embeddings.
        batch_size (int, optional): Batch size. Defaults to 16.
        mode (str): "Returns different dataloader depending on whether its "train", "valid"\
            or "predict" mode.

    Returns
    -------
        torch.utils.data.dataloader.DataLoader: Dataloader to use for training.
    """
    shuffle = mode == "train"
    dataset = ParaplumeDataset(dataset_dict=dataset_dict, antibody_embeddings=embeddings, mode=mode)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
