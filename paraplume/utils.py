"""Contain some useful functions for the project."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

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
from tqdm import tqdm


def rec_dd():
    """Return nested default_dict.

    Returns
    -------
        defaultdict: Dictionary where you can add as many nested keys as you wish.
    """
    return defaultdict(rec_dd)


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


def get_dim(embedding_models: list[str]) -> int:
    """Return the dimension -1 of the concatenated embedding.

    Args:
        embedding_models (List[str]): List of embeddings to use.

    Returns
    -------
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


def get_binding_residues(
    df_antibody: pd.DataFrame, df_antigen: pd.DataFrame
) -> tuple[dict[str, str], dict[str, list[float]]]:
    """Return dictionary with imgt positions mapping to aa name, and list of distances to antigen.

    Args:
        df_antibody (pd.DataFrame): Dataframe of antibody chain with columns taken from pdb file.
        df_antigen (pd.DataFrame): Dataframe of antigen chain with columns taken from pdb file.

    Returns
    -------
        position_dict, distance_dict (Tuple[Dict[str,str], Dict[str,List[float]]]): Dictionaries \
            with imgt positions mapping to aa name, and list of distances to antigen.
    """
    antibody_coords = df_antibody[["x_coord", "y_coord", "z_coord"]].astype(float).to_numpy()
    antigen_coords = df_antigen[["x_coord", "y_coord", "z_coord"]].astype(float).to_numpy()

    values = df_antibody[["residue_name", "IMGT"]].values
    distances = np.linalg.norm(antibody_coords[:, np.newaxis] - antigen_coords, axis=2)
    position_dict: dict[str, str] = {}
    distance_dict: dict[str, list[float]] = {}
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
    positions: dict[str, str],
    distances: dict[str, list[float]],
    alpha: float = 4.5,
) -> tuple[list, list, list]:
    """Create lists with label, aa, imgt position that all map.

    Args:
        positions (Dict[str,str]): Dictionary mapping IMGT position to amino acid
        distances (Dict[str, List[float]]): Dictionary mapping IMGT position to the distances of \
            antigen atoms to this aa
        alpha:float Alpha to use for computing labels.

    Returns
    -------
        Tuple[List, List, List]: Lists with label, aa, imgt position that all map
    """
    labels = []
    sequence = []
    numbers = []
    for pos, seq in positions.items():
        sequence.append(seq)
        numbers.append(pos)
        if np.min(distances[pos]) <= alpha:
            labels.append(1)
        else:
            labels.append(0)
    return labels, sequence, numbers


def save_plot(
    train_loss_list: list,
    val_loss_list: list,
    auc_list: list,
    ap_list: list,
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

    Returns
    -------
        pd.DataFrame: A DataFrame containing the atomic coordinates and metadata, removed of it's
        hydrogen atoms with one row per atom.
    """
    atomic_df = PandasPdb().read_pdb(pdb_path).df["ATOM"].query("element_symbol!='H'")
    atomic_df["IMGT"] = atomic_df["residue_number"].astype(str) + atomic_df["insertion"].astype(str)
    return atomic_df


def youdens_index(targets: np.ndarray, predictions: list[int]) -> float:
    """Compute Youden index given two vectors.

    Args:
        targets (np.array): Targets
        predictions (np.array): Predictions

    Returns
    -------
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
    return structlog.get_logger()


def get_metrics(
    all_outputs: np.ndarray, all_targets: np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
    """Get AP, AUC, F1 score, MCC score, and the best thresholds for F1 and MCC scores.

    Args:
        all_outputs (List): Outputs of the model
        all_targets (List): Labels

    Returns
    -------
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


def build_dictionary(
    pdb_dataframe: pd.DataFrame,
    pdb_folder_path: Path = Path("/home/gathenes/all_structures/imgt_renumbered_expanded"),
) -> dict[str, dict[str, Any]]:
    """Transform dataframe with pdb codes, heavy and light chains into dictionary.

    Args:
        pdb_dataframe (pd.DataFrame): Dataframe with pdb codes and heavy and light chain names.
        pdb_folder_path (Path): Folder path from which to construct ground truth.

    Returns
    -------
        Dict[str, Dict[str, Any]]: Dictionary with indices mapping to heavy and light lists of \
            matching imgt numbers, sequences and labels.
    """
    dataset_dict = rec_dd()
    for index in tqdm(range(len(pdb_dataframe))):
        # get pdb codes and chain names
        pdb_code = pdb_dataframe.iloc[index]["pdb"]
        h_id = pdb_dataframe.iloc[index]["Hchain"]  # noqa: F841
        l_id = pdb_dataframe.iloc[index]["Lchain"]  # noqa: F841
        antigen_id = pdb_dataframe.iloc[index]["antigen_chain"]

        # load dataframe
        if not (pdb_folder_path / Path(f"{pdb_code}.pdb")).exists():
            raise ValueError(f"{pdb_code} not in {pdb_folder_path.as_posix()}")
        df_pdb = read_pdb_to_dataframe(pdb_folder_path / Path(f"{pdb_code}.pdb"))
        # Get each dataframe for each chain type
        df_chain_heavy = df_pdb.query("chain_id == @h_id and residue_number<129")
        df_chain_light = df_pdb.query("chain_id == @l_id and residue_number<128")
        antigen_ids = antigen_id.split(";")  # noqa: F841
        df_chain_antigen = df_pdb.query("chain_id.isin(@antigen_ids)")
        if len(df_chain_antigen) == 0:
            raise ValueError(f"Empty antigen, please check pdb {pdb_code}")

        if len(df_chain_heavy) > 0:
            position_dict_heavy, distance_dict_heavy = get_binding_residues(
                df_chain_heavy, df_chain_antigen
            )
            labels_heavy_4_5, sequence_heavy, numbers_heavy = get_labels(
                position_dict_heavy, distance_dict_heavy, alpha=4.5
            )
            distances_heavy = [np.min(distance_dict_heavy[each]) for each in numbers_heavy]
            dataset_dict[index]["H_id distances"] = distances_heavy
            dataset_dict[index]["H_id numbers"] = numbers_heavy
            dataset_dict[index]["H_id sequence"] = "".join(sequence_heavy)
            dataset_dict[index]["H_id labels 4.5"] = labels_heavy_4_5
            for alpha in [3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5]:
                labels_heavy, _, _ = get_labels(
                    position_dict_heavy, distance_dict_heavy, alpha=alpha
                )
                dataset_dict[index][f"H_id labels {alpha}"] = labels_heavy
        else:
            dataset_dict[index]["H_id distances"] = []
            dataset_dict[index]["H_id numbers"] = []
            dataset_dict[index]["H_id sequence"] = ""
            dataset_dict[index]["H_id labels 4.5"] = []
            for alpha in [3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5]:
                dataset_dict[index][f"H_id labels {alpha}"] = []

        if len(df_chain_light) > 0:
            position_dict_light, distance_dict_light = get_binding_residues(
                df_chain_light, df_chain_antigen
            )
            labels_light_4_5, sequence_light, numbers_light = get_labels(
                position_dict_light, distance_dict_light, alpha=4.5
            )
            distances_light = [np.min(distance_dict_light[each]) for each in numbers_light]
            dataset_dict[index]["L_id distances"] = distances_light
            dataset_dict[index]["L_id numbers"] = numbers_light
            dataset_dict[index]["L_id sequence"] = "".join(sequence_light)
            dataset_dict[index]["L_id labels 4.5"] = labels_light_4_5
            for alpha in [3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5]:
                labels_light, _, _ = get_labels(
                    position_dict_light,
                    distance_dict_light,
                    alpha=alpha,
                )
                dataset_dict[index][f"L_id labels {alpha}"] = labels_light
        else:
            dataset_dict[index]["L_id distances"] = []
            dataset_dict[index]["L_id numbers"] = []
            dataset_dict[index]["L_id sequence"] = ""
            dataset_dict[index]["L_id labels 4.5"] = []
            for alpha in [3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5]:
                dataset_dict[index][f"L_id labels {alpha}"] = []

        dataset_dict[index]["pdb_code"] = pdb_code

    return dataset_dict


def get_embedding(
    embedding: torch.Tensor, embedding_models: list[str], heavy: int, light: int
) -> torch.Tensor:
    """Return tensor of embedding given lengths of heavy and light chains and embeddings to use.

    Args:
        embedding (torch.Tensor): Total embedding in which to do selection.
        embedding_models (List[str]): List of embeddings to use.
        heavy (int): Heavy chain length.
        light (int): Light chain length.

    Returns
    -------
        torch.Tensor: Embedding tensor to be used by the model.
    """
    embedding_coords_aa = {
        "ablang2": list(range(1, heavy + 1)) + list(range(heavy + 4, heavy + light + 4)),
        "igT5": list(range(1, heavy + 1)) + list(range(heavy + 2, heavy + light + 2)),
        "igbert": list(range(1, heavy + 1)) + list(range(heavy + 2, heavy + light + 2)),
        "esm": list(range(1, heavy + light + 1)),
        "antiberty": list(range(1, heavy + light + 1)),
        "prot-t5": list(range(heavy + light)),
    }
    embedding_coords_embpos = {
        "ablang2": list(range(2048, 2528)),
        "igT5": list(range(1024, 2048)),
        "igbert": list(range(1024)),
        "esm": list(range(2528, 3808)),
        "antiberty": list(range(3808, 4320)),
        "prot-t5": list(range(4320, 5344)),
    }
    emb_list = []
    for emb_model in embedding_models:
        ran_aa = embedding_coords_aa[emb_model]
        ran_embpos = embedding_coords_embpos[emb_model]
        emb_list.append(embedding[ran_aa][:, ran_embpos])
    return torch.cat(emb_list, dim=1)

def get_device(gpu=0):
    if torch.cuda.is_available():
        if gpu < torch.cuda.device_count() and gpu>=0:
            return torch.device(f"cuda:{gpu}")
        else:
            print(f"Warning: GPU index {gpu} not available. Falling back to CPU.")
    return torch.device("cpu")

log = get_logger()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0, path=Path("./checkpoint.pt"), best_score=0):
        """Initialize class.

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

    def __call__(self, val_loss: float, model: torch.nn.Sequential):
        """Save weights if val_loss increased for less than 'patience' epochs, +1 counter otherwise.

        Args:
            val_loss (float): Value to maximize on validation set.
            model (torch.nn.Sequential): Model that is being trained.
        """
        score = -val_loss
        if score <= self.best_score + self.delta:
            self.counter += 1
            log.info("EarlyStopping counter", counter=self.counter, patience=self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            log.info(
                "AP AUC increased", before=-round(self.val_loss_min, 6), after=-round(val_loss, 6)
            )
            self.val_loss_min = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path.as_posix())
