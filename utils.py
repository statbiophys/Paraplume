"""Contain some useful functions for the project."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from biopandas.pdb import PandasPdb
from tqdm import tqdm


# pylint: disable-msg=too-many-locals
# pylint: disable=unused-variable
def rec_dd():
    """Return nested default_dict.

    Returns:
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


def get_embedding(
    embedding: torch.Tensor, embedding_models: List[str], heavy: int, light: int
) -> torch.Tensor:
    """Return tensor of embedding given lengths of heavy and light chains and embeddings to use.

    Args:
        embedding (torch.Tensor): Total embedding in which to do selection.
        embedding_models (List[str]): List of embeddings to use.
        heavy (int): Heavy chain length.
        light (int): Light chain length.

    Returns:
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
    emb = torch.cat(emb_list, dim=1)
    return emb


def get_binding_residues(
    df_antibody: pd.DataFrame, df_antigen: pd.DataFrame
) -> Tuple[Dict[str, str], Dict[str, List[float]]]:
    """Return dictionary with imgt positions mapping to aa name, and list of distances to antigen.

    Args:
        df_antibody (pd.DataFrame): Dataframe of antibody chain with columns taken from pdb file.
        df_antigen (pd.DataFrame): Dataframe of antigen chain with columns taken from pdb file.

    Returns:
        position_dict, distance_dict(Tuple[Dict[str,str], Dict[str,List[float]]]): Dictionaries \
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


def build_dictionary(
    pdbs_and_chain: pd.DataFrame,
    pdb_folder_path: Path = Path("/home/gathenes/all_structures/imgt_renumbered_expanded"),
) -> Dict[str, Dict[str, Any]]:
    """Transform dataframe with pdb codes and heavy and light chain names into Dictionary with \
        indices mapping to heavy and light lists of matching imgt numbers, sequences and labels.

    Args:
        pdbs_and_chain (pd.DataFrame): Dataframe with pdb codes and heavy and light chain names.
        pdb_folder_path (Path): Folder path from which to construct ground truth.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with indices mapping to heavy and light lists of \
            matching imgt numbers, sequences and labels.
    """
    dataset_dict = rec_dd()
    for index in tqdm(range(len(pdbs_and_chain))):
        # get pdb codes and chain names
        pdb_code = pdbs_and_chain.iloc[index]["pdb"]
        h_id = pdbs_and_chain.iloc[index]["Hchain"]
        l_id = pdbs_and_chain.iloc[index]["Lchain"]
        antigen_id = pdbs_and_chain.iloc[index]["antigen_chain"]

        # load dataframe
        df_pdb = read_pdb_to_dataframe(pdb_folder_path / Path(f"{pdb_code}.pdb"))

        # Get each dataframe for each chain type
        df_chain_heavy = df_pdb.query("chain_id == @h_id and residue_number<129")
        df_chain_light = df_pdb.query("chain_id == @l_id and residue_number<128")
        antigen_ids = antigen_id.split(";")
        df_chain_antigen = df_pdb.query("chain_id.isin(@antigen_ids)")

        # Get binding residues
        position_dict_heavy, distance_dict_heavy = get_binding_residues(
            df_chain_heavy, df_chain_antigen
        )
        labels_heavy_4_5, sequence_heavy, numbers_heavy = get_labels(
            position_dict_heavy, distance_dict_heavy, alpha=4.5
        )
        position_dict_light, distance_dict_light = get_binding_residues(
            df_chain_light, df_chain_antigen
        )
        labels_light_4_5, sequence_light, numbers_light = get_labels(
            position_dict_light, distance_dict_light, alpha=4.5
        )

        # get distances
        distances_heavy = [np.min(distance_dict_heavy[each]) for each in numbers_heavy]
        distances_light = [np.min(distance_dict_light[each]) for each in numbers_light]

        dataset_dict[index]["h_id distances"] = distances_heavy
        dataset_dict[index]["l_id distances"] = distances_light

        dataset_dict[index]["h_id numbers"] = numbers_heavy
        dataset_dict[index]["l_id numbers"] = numbers_light

        dataset_dict[index]["h_id sequence"] = "".join(sequence_heavy)
        dataset_dict[index]["l_id sequence"] = "".join(sequence_light)

        dataset_dict[index]["h_id labels 4.5"] = labels_heavy_4_5
        dataset_dict[index]["l_id labels 4.5"] = labels_light_4_5

        for alpha in [3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5]:
            labels_heavy, _, _ = get_labels(position_dict_heavy, distance_dict_heavy, alpha=alpha)
            labels_light, _, _ = get_labels(
                position_dict_light,
                distance_dict_light,
                alpha=alpha,
            )
            dataset_dict[index][f"h_id labels {alpha}"] = labels_heavy
            dataset_dict[index][f"l_id labels {alpha}"] = labels_light

        # Save pdb code
        dataset_dict[index]["pdb_code"] = pdb_code

    return dataset_dict


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
        hydrogen atomswith one row per atom.
    """
    atomic_df = PandasPdb().read_pdb(pdb_path).df["ATOM"].query("element_symbol!='H'")
    atomic_df["IMGT"] = atomic_df["residue_number"].astype(str) + atomic_df["insertion"].astype(str)
    return atomic_df


def get_other_labels(
    dataset_dict: Dict[str, Dict[str, Any]], index: int, alphas: Optional[List] = None
) -> List[torch.Tensor]:
    """Return list of tensors of padded labels for different alphas.

    Args:
        dataset_dict (Dict[str, Dict[str, Any]]): Dictionary mapping indices to positions, \
            imgt numbers and labels.
        index (int): Index of dictionary.
        alphas (Optional[List]): List of alphas to use for multi objective \
            optimization. Defaults to None.

    Returns:
        List[torch.Tensor]: List of padded labels.
    """
    labels_list: List[torch.Tensor] = []
    if alphas is None:
        return labels_list
    for alpha in alphas:
        labels_heavy = dataset_dict[str(index)][f"h_id labels {alpha}"]
        labels_light = dataset_dict[str(index)][f"l_id labels {alpha}"]
        labels_paired = labels_heavy + labels_light
        labels_padded = torch.FloatTensor(
            F.pad(
                torch.FloatTensor(labels_paired),
                (0, 280 - len(torch.FloatTensor(labels_paired))),
                "constant",
                0,
            )
        )
        labels_list.append(labels_padded)
    return labels_list
