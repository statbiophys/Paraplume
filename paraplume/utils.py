"""Contain some useful functions for the project."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog
import torch
from biopandas.pdb import PandasPdb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

# pylint: disable-msg=too-many-locals
# pylint: disable=unused-variable


amino_acid_dict = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def get_dim(embedding_models: List[str]) -> int:
    """Return the dimension -1 of the concatenated embedding.

    Args:
        embedding_models (List[str]): List of embeddings to use.

    Returns:
        int: Dimension -1.
    """
    embedding_dims = {
        "ablang2": 480,
        "igT5": 1024,
        "igbert": 1024,
        "esm": 1280,
        "antiberty": 512,
        "prot-t5": 1024,
    }
    dim = 0
    for emb_model in embedding_models:
        dim += embedding_dims[emb_model]
    return dim


def get_node_embedding(
    embedding: torch.Tensor,
    index: int,
    embedding_models: List[str],
    chain: str,
) -> torch.Tensor:
    """Return tensor of embedding given node index, chain and embeddings list used.

    Args:
        embedding (torch.Tensor): Total embedding in which to do selection.
        index (int): Node index.
        embedding_models (List[str]): List of embeddings to use.
        chain (str): Chain to use. Defaults to "heavy".

    Returns:
        torch.Tensor: _description_
    """
    embedding_coords_embpos = {
        "ablang2": list(range(2048, 2528)),
        "igT5": list(range(1024, 2048)),
        "igbert": list(range(1024)),
        "esm": list(range(2528, 3808)),
        "antiberty": list(range(3808, 4320)),
        "prot-t5": list(range(4320, 5344)),
    }
    embedding_coords_aa = {
        "heavy": {
            "ablang2": 1 + index,
            "igT5": 1 + index,
            "igbert": 1 + index,
            "esm": 1 + index,
            "antiberty": 1 + index,
            "prot-t5": index,
        },
        "light": {
            "ablang2": 4 + index,
            "igT5": 2 + index,
            "igbert": 2 + index,
            "esm": 1 + index,
            "antiberty": 1 + index,
            "prot-t5": index,
        },
    }
    emb_list = []
    for emb_model in embedding_models:
        ran_aa = embedding_coords_aa[chain][emb_model]
        ran_embpos = embedding_coords_embpos[emb_model]
        emb_list.append(embedding[ran_aa][ran_embpos])
    emb = torch.cat(emb_list, dim=0)
    return emb


def get_binding_residues(
    df_antibody: pd.DataFrame, df_antigen: pd.DataFrame
) -> Tuple[Dict[str, str], Dict[str, List[float]]]:
    """Return dictionary with imgt positions mapping to aa name, and list of distances to antigen.

    Args:
        df_antibody (pd.DataFrame): Dataframe of antibody chain with columns taken from pdb file.
        df_antigen (pd.DataFrame): Dataframe of antigen chain with columns taken from pdb file.

    Returns:
        position_dict, distance_dict (Tuple[Dict[str,str], Dict[str,List[float]]]): Dictionaries \
            with imgt positions mapping to aa name, and list of distances to antigen.
    """
    antibody_coords = df_antibody[["x_coord", "y_coord", "z_coord"]].astype(float).to_numpy()
    antigen_coords = df_antigen[["x_coord", "y_coord", "z_coord"]].astype(float).to_numpy()

    values = df_antibody[["residue_name", "IMGT"]].values
    distances = np.linalg.norm(antibody_coords[:, np.newaxis] - antigen_coords, axis=2)
    position_dict: Dict[str, str] = {}
    distance_dict: Dict[str, List[float]] = {}
    for row_ab, (res_name, res_num) in enumerate(values):
        if res_num == "nan":
            raise ValueError(res_num)
        position_dict[res_num] = amino_acid_dict[res_name]
        if res_num not in distance_dict:
            distance_dict[res_num] = [np.min(distances[row_ab, :])]
        else:
            distance_dict[res_num].append(np.min(distances[row_ab, :]))
    return position_dict, distance_dict


def get_labels(
    positions: Dict[str, str],
    distances: Dict[str, List[float]],
    alpha: float = 4.5,
) -> Tuple[List, List, List]:
    """Create lists with label, aa, imgt position that all map.

    Args:
        positions (Dict[str,str]): Dictionary mapping IMGT position to amino acid
        distances (Dict[str, List[float]]): Dictionary mapping IMGT position to the distances of \
            antigen atoms to this aa
        alpha:float Alpha to use for computing labels.

    Returns:
        Tuple[List, List, List]: Lists with label, aa, imgt position that all map
    """
    labels = []
    sequence = []
    numbers = []
    for pos in positions:
        sequence.append(positions[pos])
        numbers.append(pos)
        if np.min(distances[pos]) <= alpha:
            labels.append(1)
        else:
            labels.append(0)
    return labels, sequence, numbers

def save_plot(
    train_loss_list: List,
    val_loss_list: List,
    auc_list: List,
    ap_list: List,
    save_plot_path: Path,
) -> None:
    """Save different training plots.

    Args:
        train_loss_list (List): Training loss list.
        val_loss_list (List): Validation loss list.
        auc_list (List): Roc auc validation list.
        ap_list (List): Pr auc validation list.
        save_plot_path (Path): Where to save the plots.
    """
    n_epochs = len(train_loss_list)

    # Create a figure with 2x2 subplots
    _, axs = plt.subplots(2, 2, figsize=(10, 10))

    # First subplot for Train Loss
    axs[0, 0].plot(range(1, n_epochs + 1), train_loss_list)
    axs[0, 0].set_xlabel("num_epochs")
    axs[0, 0].set_ylabel("Train Loss")
    axs[0, 0].set_title("Train Loss")

    # Second subplot for val Loss
    axs[0, 1].plot(range(1, n_epochs + 1), val_loss_list)
    axs[0, 1].set_xlabel("num_epochs")
    axs[0, 1].set_ylabel("val Loss")
    axs[0, 1].set_title("val Loss")

    # Third subplot for AUC
    axs[1, 0].plot(range(1, n_epochs + 1), auc_list)
    axs[1, 0].set_xlabel("num_epochs")
    axs[1, 0].set_ylabel("ROC AUC")
    axs[1, 0].set_title("ROC AUC")

    # Fourth subplot for AP
    axs[1, 1].plot(range(1, n_epochs + 1), ap_list)
    axs[1, 1].set_xlabel("num_epochs")
    axs[1, 1].set_ylabel("ROC AP")
    axs[1, 1].set_title("ROC AP")

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.savefig(save_plot_path)


def read_pdb_to_dataframe(
    pdb_path: Path,
) -> pd.DataFrame:
    """
    Read a PDB file, and return a Pandas DataFrame containing the atomic coordinates and metadata.

    Args:
        pdb_path (str, optional): Path to a local PDB file to read. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the atomic coordinates and metadata, removed of it's
        hydrogen atoms with one row per atom.
    """
    atomic_df = PandasPdb().read_pdb(pdb_path).df["ATOM"].query("element_symbol!='H'")
    atomic_df["IMGT"] = atomic_df["residue_number"].astype(str) + atomic_df["insertion"].astype(str)
    return atomic_df


def youdens_index(targets: np.ndarray, predictions: List[int]) -> float:
    """Compute Youden index given two vectors.

    Args:
        targets (np.array): Targets
        predictions (np.array): Predictions

    Returns:
        float: Youden index
    """
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity + specificity - 1


def get_logger():
    """Return logger."""
    renderer = structlog.dev.ConsoleRenderer(sort_keys=False)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
    )
    log = structlog.get_logger()
    return log

def get_metrics(
    all_outputs: np.ndarray, all_targets: np.ndarray
) -> Tuple[float, float, float, float, float, float, float]:
    """Get AP, AUC, F1 score, MCC score, and the best thresholds for F1 and MCC scores.

    Args:
        all_outputs (List): Outputs of the model
        all_targets (List): Labels

    Returns:
        Tuple[float,float, float,float,float,float,float]: AUC, AP, Youden threshold \
            best_mcc_threshold, best_f1_threshold, best f1, best mcc
    """
    auc = roc_auc_score(all_targets, all_outputs)
    ap = average_precision_score(all_targets, all_outputs)
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    mcc_scores = []
    youden_indices = []

    # Iterate through thresholds and calculate F1 and MCC scores
    for threshold in thresholds:
        all_predictions = [1 if x >= threshold else 0 for x in all_outputs]
        f1_scores.append(f1_score(all_targets, all_predictions))
        mcc_scores.append(matthews_corrcoef(all_targets, all_predictions))
        youden_indices.append(youdens_index(all_targets, all_predictions))

    # Find the threshold that maximizes Youden's Index
    max_index = np.argmax(youden_indices)
    best_threshold = thresholds[max_index]
    best_validation_mcc_threshold = thresholds[np.argmax(mcc_scores)]
    best_validation_f1_threshold = thresholds[np.argmax(f1_scores)]
    return (
        auc,
        ap,
        best_threshold,
        best_validation_mcc_threshold,
        best_validation_f1_threshold,
        np.max(f1_scores),
        np.max(mcc_scores),
    )
