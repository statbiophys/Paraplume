from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    antibody_coords = df_antibody[["x", "y", "z"]].astype(float).to_numpy()
    antigen_coords = df_antigen[["x", "y", "z"]].astype(float).to_numpy()

    # Compute pairwise distances

    values = df_antibody[["AA", "Res_Num"]].values
    distances = np.linalg.norm(antibody_coords[:, np.newaxis] - antigen_coords, axis=2)
    amino_dict = {"positions": {}, "distances": {}}

    for row_ab, (AA, Res_Num) in enumerate(values):
        # print(AA,Res_Num)
        if Res_Num=="nan":
            raise ValueError(Res_Num)
        amino_dict["positions"][Res_Num] = amino_acid_dict[AA]
        if Res_Num not in amino_dict["distances"]:
            amino_dict["distances"][Res_Num] = [np.min(distances[row_ab, :])]

        else:
            amino_dict["distances"][Res_Num].append(np.min(distances[row_ab, :]))
    return amino_dict

def get_labels(
    positions:Dict[int,str], distances:Dict[int, List], percentile:int=0, alpha:float=4.5
)->Tuple[List,List,List]:
    """Create lists with label, aa, imgt position that all map.

    Args:
        positions (Dict[int,str]): Dictionary mapping IMGT position to amino acid
        distances (Dict[int, List]): Dictionary mapping IMGT position to the distances of antigen \
            atoms to this aa
        percentile (int, optional): Percentile to use to get the distance to antigen. Defaults to 0.
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
        if np.percentile(distances[pos], percentile)<= alpha:
            labels.append(1)
        else:
            labels.append(0)
    return labels, sequence, numbers

def build_dictionary(pdbs_and_chain:pd.DataFrame)->Dict:
    """Transform dataframe with pdb codes and heavy and light chain names into Dictionary with \
        indices mapping to heavy and light lists of matching imgt numbers, sequences and labels.

    Args:
        pdbs_and_chain (pd.DataFrame): Dataframe with pdb codes and heavy and light chain names.

    Returns:
        Dict: Dictionary with indices mapping to heavy and light lists of matching imgt numbers, \
            sequences and labels.
    """
    dataset_dict = rec_dd()
    for index in tqdm(range(len(pdbs_and_chain))):
        pdb_code = pdbs_and_chain.iloc[index]["pdb"]

        H_id = pdbs_and_chain.iloc[index]["Hchain"]
        L_id = pdbs_and_chain.iloc[index]["Lchain"]
        antigen_id = pdbs_and_chain.iloc[index]["antigen_chain"]
        df = format_pdb(f"/home/gathenes/all_structures/imgt/{pdb_code}.pdb")

        df_chain_heavy = df.query("Chain == @H_id")
        df_chain_light = df.query("Chain == @L_id")
        antigen_ids=antigen_id.split(";")
        df_chain_antigen = df.query("Chain.isin(@antigen_ids)")
        abr_dict_heavy = get_binding_residues(df_chain_heavy, df_chain_antigen)
        labels_heavy_4_5, sequence_heavy, numbers_heavy = get_labels(abr_dict_heavy["positions"], abr_dict_heavy["distances"], alpha=4.5)
        abr_dict_light = get_binding_residues(df_chain_light, df_chain_antigen)
        labels_light_4_5, sequence_light, numbers_light = get_labels(abr_dict_light["positions"], abr_dict_light["distances"], alpha=4.5)

        dataset_dict[index]["pdb_code"]=pdb_code
        inverse_number_heavy = {each : i for i,each in enumerate(numbers_heavy)}
        inverse_number_light = {each : i for i,each in enumerate(numbers_light)}
        left, right=1, 128
        while str(left) not in inverse_number_heavy :
            left+=1
        while str(right) not in inverse_number_heavy:
            right-=1
        heavy_left, heavy_right = inverse_number_heavy[str(left)], inverse_number_heavy[str(right)]
        left, right=1, 128
        while str(left) not in inverse_number_light :
            left+=1
        while str(right) not in inverse_number_light:
            right-=1
        light_left, light_right = inverse_number_light[str(left)], inverse_number_light[str(right)]

        labels_heavy_4_5, sequence_heavy, numbers_heavy = labels_heavy_4_5[heavy_left:heavy_right+1], sequence_heavy[heavy_left:heavy_right+1], numbers_heavy[heavy_left:heavy_right+1]
        labels_light_4_5, sequence_light, numbers_light = labels_light_4_5[light_left:light_right+1], sequence_light[light_left:light_right+1], numbers_light[light_left:light_right+1]

        dataset_dict[index]["H_id numbers"]=numbers_heavy
        dataset_dict[index]["L_id numbers"]=numbers_light

        dataset_dict[index]["H_id sequence"] = "".join(sequence_heavy)
        dataset_dict[index]["H_id labels 4.5"] = labels_heavy_4_5
        dataset_dict[index]["L_id sequence"] = "".join(sequence_light)
        dataset_dict[index]["L_id labels 4.5"] = labels_light_4_5

        for alpha in [3,3.5,4,5,5.5,6,6.5,7,7.5]:
            labels_heavy, _, _ = get_labels(abr_dict_heavy["positions"], abr_dict_heavy["distances"], alpha=alpha)
            labels_light, _, _ = get_labels(abr_dict_light["positions"], abr_dict_light["distances"], alpha=alpha)
            labels_heavy = labels_heavy[heavy_left:heavy_right+1]
            labels_light = labels_light[light_left:light_right+1]
            dataset_dict[index][f"H_id labels {alpha}"] = labels_heavy
            dataset_dict[index][f"L_id labels {alpha}"] = labels_light
    return dataset_dict


    return dataset_dict

def format_pdb(pdb_file):
    '''
    Process pdb file into pandas df

    Original author: Alissa Hummer

    :param pdb_file: file path of .pdb file to convert
    :returns: df with atomic level info
    '''

    pd.options.mode.chained_assignment = None
    pdb_whole = pd.read_csv(pdb_file,header=None,delimiter='\t')
    pdb_whole.columns = ['pdb']
    pdb = pdb_whole[pdb_whole['pdb'].str.startswith('ATOM')]
    pdb['Atom_Num'] = pdb['pdb'].str[6:11].copy()
    pdb['Atom_Name'] = pdb['pdb'].str[11:16].copy()
    pdb['AA'] = pdb['pdb'].str[17:20].copy()
    pdb['Chain'] = pdb['pdb'].str[20:22].copy()
    pdb['Res_Num'] = pdb['pdb'].str[22:27].copy().str.strip()
    pdb['x'] = pdb['pdb'].str[27:38].copy()
    pdb['y'] = pdb['pdb'].str[38:46].copy()
    pdb['z'] = pdb['pdb'].str[46:54].copy()#
    pdb['Atom_type'] = pdb['pdb'].str[77].copy()
    pdb.drop('pdb',axis=1,inplace=True)
    pdb.replace({' ':''}, regex=True, inplace=True)
    pdb.reset_index(inplace=True)
    pdb.drop('index',axis=1,inplace=True)

    # remove H atoms from our data (interested in heavy atoms only)
    pdb = pdb[pdb['Atom_type']!='H']

    return pdb

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
