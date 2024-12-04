from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from biopandas.pdb import PandasPdb
from tqdm import tqdm


def rec_dd():
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


def get_binding_residues(
    df_antibody: pd.DataFrame, df_antigen: pd.DataFrame
) -> Dict[str, Dict[int, str]]:
    """Return dictionary with imgt positions mapping to aa name, and list of distances to antigen.

    Args:
        df_antibody (pd.DataFrame): Dataframe of antibody chain with columns taken from pdb file.
        df_antigen (pd.DataFrame): Dataframe of antigen chain with columns taken from pdb file.

    Returns:
        Dict[str, Dict[int, str]]: Dictionary with imgt positions mapping to aa name, and list \
            of distances to antigen
    """

    antibody_coords = df_antibody[["x_coord", "y_coord", "z_coord"]].astype(float).to_numpy()
    antigen_coords = df_antigen[["x_coord", "y_coord", "z_coord"]].astype(float).to_numpy()

    # Compute pairwise distances

    values = df_antibody[["residue_name", "IMGT"]].values
    distances = np.linalg.norm(antibody_coords[:, np.newaxis] - antigen_coords, axis=2)
    amino_dict = {"positions": {}, "distances": {}}

    for row_ab, (res_name, res_num) in enumerate(values):
        if res_num == "nan":
            raise ValueError(res_num)
        amino_dict["positions"][res_num] = amino_acid_dict[res_name]
        if res_num not in amino_dict["distances"]:
            amino_dict["distances"][res_num] = [np.min(distances[row_ab, :])]
        else:
            amino_dict["distances"][res_num].append(np.min(distances[row_ab, :]))
    return amino_dict


def get_labels(
    positions: Dict[int, str],
    distances: Dict[int, List],
    alpha: float = 4.5,
) -> Tuple[List, List, List]:
    """Create lists with label, aa, imgt position that all map.

    Args:
        positions (Dict[int,str]): Dictionary mapping IMGT position to amino acid
        distances (Dict[int, List]): Dictionary mapping IMGT position to the distances of antigen \
            atoms to this aa
        alpha (float, optional): Cut off to use to label aa as paratope. Defaults to 4.5.

    Returns:
        Tuple[List,List,List]: Lists with label, aa, imgt position that all map
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
) -> Dict:
    """Transform dataframe with pdb codes and heavy and light chain names into Dictionary with \
        indices mapping to heavy and light lists of matching imgt numbers, sequences and labels.

    Args:
        pdbs_and_chain (pd.DataFrame): Dataframe with pdb codes and heavy and light chain names.
        pdb_folder_path (Path): Folder path from which to construct ground truth.

    Returns:
        Dict: Dictionary with indices mapping to heavy and light lists of matching imgt numbers, \
            sequences and labels.
    """
    dataset_dict = rec_dd()
    for index in tqdm(range(len(pdbs_and_chain))):

        # get pdb codes and chain names
        pdb_code = pdbs_and_chain.iloc[index]["pdb"]
        H_id = pdbs_and_chain.iloc[index]["Hchain"]
        L_id = pdbs_and_chain.iloc[index]["Lchain"]
        antigen_id = pdbs_and_chain.iloc[index]["antigen_chain"]

        # load dataframe
        df = read_pdb_to_dataframe(pdb_folder_path / Path(f"{pdb_code}.pdb"))

        # Get each dataframe for each chain type
        df_chain_heavy = df.query("chain_id == @H_id and residue_number<129")
        df_chain_light = df.query("chain_id == @L_id and residue_number<128")
        antigen_ids = antigen_id.split(";")
        df_chain_antigen = df.query("chain_id.isin(@antigen_ids)")

        # Get binding residues
        abr_dict_heavy = get_binding_residues(df_chain_heavy, df_chain_antigen)
        labels_heavy_4_5, sequence_heavy, numbers_heavy = get_labels(
            abr_dict_heavy["positions"], abr_dict_heavy["distances"], alpha=4.5
        )
        abr_dict_light = get_binding_residues(df_chain_light, df_chain_antigen)
        labels_light_4_5, sequence_light, numbers_light = get_labels(
            abr_dict_light["positions"], abr_dict_light["distances"], alpha=4.5
        )

        # get distances
        distances_heavy = [
            np.min(abr_dict_heavy["distances"][each])
            for each in numbers_heavy
        ]
        distances_light = [
            np.min(abr_dict_light["distances"][each])
            for each in numbers_light
        ]


        dataset_dict[index]["H_id distances"] = distances_heavy
        dataset_dict[index]["L_id distances"] = distances_light

        dataset_dict[index]["H_id numbers"] = numbers_heavy
        dataset_dict[index]["L_id numbers"] = numbers_light

        dataset_dict[index]["H_id sequence"] = "".join(sequence_heavy)
        dataset_dict[index]["L_id sequence"] = "".join(sequence_light)

        dataset_dict[index]["H_id labels 4.5"] = labels_heavy_4_5
        dataset_dict[index]["L_id labels 4.5"] = labels_light_4_5

        for alpha in [3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5]:
            labels_heavy, _, _ = get_labels(
                abr_dict_heavy["positions"], abr_dict_heavy["distances"], alpha=alpha
            )
            labels_light, _, _ = get_labels(
                abr_dict_light["positions"], abr_dict_light["distances"], alpha=alpha
            )
            dataset_dict[index][f"H_id labels {alpha}"] = labels_heavy
            dataset_dict[index][f"L_id labels {alpha}"] = labels_light

        # Save pdb code
        dataset_dict[index]["pdb_code"] = pdb_code

    return dataset_dict



def save_plot(train_loss_list, val_loss_list, auc_list, ap_list, save_plot_path):
    n_epochs = len(train_loss_list)

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

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
    axs[1, 0].set_ylabel("AUC")
    axs[1, 0].set_title("AUC")

    # Fourth subplot for AP
    axs[1, 1].plot(range(1, n_epochs + 1), ap_list)
    axs[1, 1].set_xlabel("num_epochs")
    axs[1, 1].set_ylabel("AP")
    axs[1, 1].set_title("Average Precision (AP)")

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.savefig(save_plot_path)


def read_pdb_to_dataframe(
    pdb_path: Optional[str] = None,
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
    atomic_df["IMGT"]=atomic_df["residue_number"].astype(str)+atomic_df["insertion"].astype(str)
    return atomic_df

def get_other_labels(dataset_dict, index, alphas=Optional[List]):
    labels_list = []
    if alphas is None:
        return labels_list
    for alpha in alphas:
        labels_heavy = dataset_dict[str(index)][f"H_id labels {alpha}"]
        labels_light = dataset_dict[str(index)][f"L_id labels {alpha}"]
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
