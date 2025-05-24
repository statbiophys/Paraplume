"""Contain some useful functions for the project."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from paraplume.utils import get_binding_residues, get_labels, read_pdb_to_dataframe


# pylint: disable-msg=too-many-locals
# pylint: disable=unused-variable
def rec_dd():
    """Return nested default_dict.

    Returns:
        defaultdict: Dictionary where you can add as many nested keys as you wish.
    """
    return defaultdict(rec_dd)

def process_embedding_single(
    llm:str, emb:torch.Tensor, chain_length: int
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
        "ablang2": list(range(1, chain_length + 1)),
        "igT5": list(range(1, chain_length + 1)),
        "igbert": list(range(1, chain_length + 1)),
        "esm": list(range(1, chain_length + 1)),
        "antiberty": list(range(1, chain_length + 1)),
        "prot-t5": list(range(chain_length)),
    }
    ran_aa = embedding_coords_aa[llm]
    return emb[ran_aa,:]

def get_embedding_single(
    embedding: torch.Tensor, embedding_models: List[str], chain_length: int
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
        "ablang2": list(range(1, chain_length + 1)),
        "igT5": list(range(1, chain_length + 1)),
        "igbert": list(range(1, chain_length + 1)),
        "esm": list(range(1, chain_length + 1)),
        "antiberty": list(range(1, chain_length + 1)),
        "prot-t5": list(range(chain_length)),
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

def build_dictionary_single(
    pdb_dataframe: pd.DataFrame,
    pdb_folder_path: Path = Path("/home/gathenes/all_structures/imgt_renumbered_expanded"),
    chain:str="heavy",
) -> Dict[str, Dict[str, Any]]:
    """Transform dataframe with pdb codes and heavy and light chain names into Dictionary with \
        indices mapping to heavy and light lists of matching imgt numbers, sequences and labels.

    Args:
        pdb_dataframe (pd.DataFrame): Dataframe with pdb codes and heavy and light chain names.
        pdb_folder_path (Path): Folder path from which to construct ground truth.
        chain(str): Chain to use

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with indices mapping to heavy and light lists of \
            matching imgt numbers, sequences and labels.
    """
    dataset_dict = rec_dd()
    for index in tqdm(range(len(pdb_dataframe))):
        # get pdb codes and chain names
        pdb_code = pdb_dataframe.iloc[index]["pdb"]
        if not (pdb_folder_path / Path(f"{pdb_code}.pdb")).exists():
            raise ValueError(f"{pdb_code} not in {pdb_folder_path.as_posix()}")

        df_pdb = read_pdb_to_dataframe(pdb_folder_path / Path(f"{pdb_code}.pdb"))

        if chain=="heavy":
            id = pdb_dataframe.iloc[index]["Hchain"]
            df_chain = df_pdb.query("chain_id == @id and residue_number<129")
        else :
            id = pdb_dataframe.iloc[index]["Lchain"]
            df_chain = df_pdb.query("chain_id == @id and residue_number<128")
        antigen_id = pdb_dataframe.iloc[index]["antigen_chain"]
        antigen_ids = antigen_id.split(";")
        df_chain_antigen = df_pdb.query("chain_id.isin(@antigen_ids)")
        if len(df_chain_antigen)== 0:
            raise ValueError(f"Empty antigen, please check pdb {pdb_code}")

        # Get binding residues
        position_dict, distance_dict = get_binding_residues(
            df_chain, df_chain_antigen
        )
        labels_4_5, sequence, numbers = get_labels(
            position_dict, distance_dict, alpha=4.5
        )

        distances = [np.min(distance_dict[each]) for each in numbers]
        dataset_dict[index]["distances"] = distances
        dataset_dict[index]["numbers"] = numbers
        dataset_dict[index]["sequence"] = "".join(sequence)
        dataset_dict[index]["labels 4.5"] = labels_4_5
        for alpha in [3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5]:
            labels, _, _ = get_labels(position_dict, distance_dict, alpha=alpha)
            dataset_dict[index][f"labels {alpha}"] = labels

        # Save pdb code
        dataset_dict[index]["pdb_code"] = pdb_code
    return dataset_dict

def get_other_labels_single(
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
        labels = dataset_dict[str(index)][f"labels {alpha}"]
        labels_padded = torch.FloatTensor(
            F.pad(
                torch.FloatTensor(labels),
                (0, 285 - len(torch.FloatTensor(labels))),
                "constant",
                0,
            )
        )
        labels_list.append(labels_padded)
    return labels_list

def get_outputs_single(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    chain_length_tensor: torch.Tensor,
    model: torch.nn.Sequential,
    embedding_models_list: List,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get model outputs and labels.

    Args:
        embedding (torch.Tensor):
        labels (torch.Tensor):
        len_heavy (torch.Tensor): Lenghts of the heavy sequences for each element of the batch.
        len_light (torch.Tensor): Lenghts of the light sequences for each element of the batch.
        model (torch.nn.Sequential):
        embedding_models_list (List): List of embedding models for which to get the\
            pre-computed embeddings.

    Returns:
        Tuple[torch.Tensor,torch.Tensor]: Labels and outputs.
    """
    embedding_list = []
    label_list = []
    for i in range(chain_length_tensor.shape[-1]):
        chain_length_i = chain_length_tensor[i]
        emb = get_embedding_single(
            embedding=embedding[i],
            embedding_models=embedding_models_list,
            chain_length=chain_length_i,
        )
        embedding_list.append(emb)
        label_list.append(labels[i][: chain_length_i])
    embedding = torch.cat(embedding_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    output = model(embedding)
    return labels, output
