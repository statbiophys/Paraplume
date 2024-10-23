from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from prody import parsePDBHeader
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
    antibody_coords = df_antibody[["x_coord", "y_coord", "z_coord"]].to_numpy()
    antigen_coords = df_antigen[["x_coord", "y_coord", "z_coord"]].to_numpy()
    # Compute pairwise distances

    values = df_antibody[["residue_name", "residue_number"]].values
    distances = np.linalg.norm(antibody_coords[:, np.newaxis] - antigen_coords, axis=2)
    amino_dict = {"positions": {}, "distances": {}}

    for row_ab, (residue_name, residue_number) in enumerate(values):
        # print(residue_name,residue_number)
        if residue_number=="nan":
            raise ValueError(residue_number)
        amino_dict["positions"][residue_number] = amino_acid_dict[residue_name]
        if residue_number not in amino_dict["distances"]:
            amino_dict["distances"][residue_number] = [np.min(distances[row_ab, :])]

        else:
            amino_dict["distances"][residue_number].append(np.min(distances[row_ab, :]))
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


def read_pdb_to_dataframe(
    pdb_path: Optional[str] = None,
    model_index: int = 1,
    parse_header: bool = True,
) -> pd.DataFrame:
    """
    Read a PDB file, and return a Pandas DataFrame containing the atomic coordinates and metadata.

    Args:
        pdb_path (str, optional): Path to a local PDB file to read. Defaults to None.
        model_index (int, optional): Index of the model to extract from the PDB file, in case
            it contains multiple models. Defaults to 1.
        parse_header (bool, optional): Whether to parse the PDB header and extract metadata.
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the atomic coordinates and metadata, with one row
            per atom
    """
    atomic_df = PandasPdb().read_pdb(pdb_path)
    if parse_header:
        header = parsePDBHeader(pdb_path)
    else:
        header = None
    atomic_df = atomic_df.get_model(model_index)
    if len(atomic_df.df["ATOM"]) == 0:
        raise ValueError(f"No model found for index: {model_index}")
    return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]]), header


def remove_ids(pdbs_and_chain: pd.DataFrame)->List:
    """Remove ids when a pdb file is missing a light, heavy or antigen chain.

    Args:
        pdbs_and_chain (pd.DataFrame): Dataframe with pdb codes, and the heavy, light and antigen \
            chain name.

    Returns:
        List : Lists of ids to remove from pdbs_and_chain
    """
    removed_ids = []
    removed_pdbs=set()
    # Extract relevant columns to reduce iloc lookups inside the loop
    ids = pdbs_and_chain["index"].values
    pdb_codes = pdbs_and_chain["pdb"].values
    H_ids = pdbs_and_chain["Hchain"].values
    L_ids = pdbs_and_chain["Lchain"].values
    antigen_ids = pdbs_and_chain["antigen_chain"].values

    # Iterate over the DataFrame more efficiently
    for index in tqdm(range(len(pdbs_and_chain))):
        pdb_code = pdb_codes[index]
        H_id = H_ids[index]
        L_id = L_ids[index]
        antigen_id = antigen_ids[index]
        ind = ids[index]
        if pdb_code in removed_pdbs:
            removed_ids.append(ind)
            continue
        # Load PDB file once per iteration
        df, _ = read_pdb_to_dataframe(f"/home/gathenes/all_structures/imgt/{pdb_code}.pdb")
        df = df.query("record_name=='ATOM'")

        # Check presence of heavy, light, and antigen chains using boolean masks
        # Add to remove_ids based on the conditions

        has_light = (df['chain_id'] == L_id).any()
        if not has_light:
            removed_ids.append(ind)
            removed_pdbs.add(pdb_code)

        has_antigen = (df['chain_id'] == antigen_id).any()
        if not has_antigen:
            removed_ids.append(ind)
            removed_pdbs.add(pdb_code)


        has_heavy = (df['chain_id'] == H_id).any()
        if not has_heavy:
            removed_ids.append(ind)
            removed_pdbs.add(pdb_code)
    pdbs_and_chain=pdbs_and_chain.query("index not in @removed_ids")
    return pdbs_and_chain

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
        if pdb_code=='2ltq':
            continue
        H_id = pdbs_and_chain.iloc[index]["Hchain"]
        L_id = pdbs_and_chain.iloc[index]["Lchain"]
        antigen_id = pdbs_and_chain.iloc[index]["antigen_chain"]
        df, _ = read_pdb_to_dataframe(
            f"/home/gathenes/all_structures/imgt/{pdb_code}.pdb"
        )
        df = df.query("record_name=='ATOM' and element_symbol!='H'")
        df_chain_heavy = df.query("chain_id == @H_id")
        df_chain_light = df.query("chain_id == @L_id")
        antigen_ids=antigen_id.split(";")
        df_chain_antigen = df.query("chain_id.isin(@antigen_ids)")
        abr_dict_heavy = get_binding_residues(df_chain_heavy, df_chain_antigen)
        labels_heavy_4_5, sequence_heavy, numbers_heavy = get_labels(abr_dict_heavy["positions"], abr_dict_heavy["distances"], alpha=4.5)
        abr_dict_light = get_binding_residues(df_chain_light, df_chain_antigen)
        labels_light_4_5, sequence_light, numbers_light = get_labels(abr_dict_light["positions"], abr_dict_light["distances"], alpha=4.5)

        dataset_dict[index]["pdb_code"]=pdb_code
        inverse_number_heavy = {each : i for i,each in enumerate(numbers_heavy)}
        inverse_number_light = {each : i for i,each in enumerate(numbers_light)}
        left, right=1, 128
        while left not in inverse_number_heavy :
            left+=1
        while right not in inverse_number_heavy:
            right-=1
        heavy_left, heavy_right = inverse_number_heavy[left], inverse_number_heavy[right]
        left, right=1, 128
        while left not in inverse_number_light :
            left+=1
        while right not in inverse_number_light:
            right-=1
        light_left, light_right = inverse_number_light[left], inverse_number_light[right]

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
