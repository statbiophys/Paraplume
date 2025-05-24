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


def get_embedding_paired(
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

def process_embedding_paired(
    llm:str, emb:torch.Tensor, heavy: int, light: int
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
    ran_aa = embedding_coords_aa[llm]
    return emb[ran_aa,:]

def build_dictionary_paired(
    pdb_dataframe: pd.DataFrame,
    pdb_folder_path: Path = Path("/home/gathenes/all_structures/imgt_renumbered_expanded"),
) -> Dict[str, Dict[str, Any]]:
    """Transform dataframe with pdb codes and heavy and light chain names into Dictionary with \
        indices mapping to heavy and light lists of matching imgt numbers, sequences and labels.

    Args:
        pdb_dataframe (pd.DataFrame): Dataframe with pdb codes and heavy and light chain names.
        pdb_folder_path (Path): Folder path from which to construct ground truth.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with indices mapping to heavy and light lists of \
            matching imgt numbers, sequences and labels.
    """
    dataset_dict = rec_dd()
    for index in tqdm(range(len(pdb_dataframe))):
        # get pdb codes and chain names
        pdb_code = pdb_dataframe.iloc[index]["pdb"]
        h_id = pdb_dataframe.iloc[index]["Hchain"]
        l_id = pdb_dataframe.iloc[index]["Lchain"]
        antigen_id = pdb_dataframe.iloc[index]["antigen_chain"]

        # load dataframe
        if not (pdb_folder_path / Path(f"{pdb_code}.pdb")).exists():
            raise ValueError(f"{pdb_code} not in {pdb_folder_path.as_posix()}")
        df_pdb = read_pdb_to_dataframe(pdb_folder_path / Path(f"{pdb_code}.pdb"))
        # Get each dataframe for each chain type
        df_chain_heavy = df_pdb.query("chain_id == @h_id and residue_number<129")
        df_chain_light = df_pdb.query("chain_id == @l_id and residue_number<128")
        antigen_ids = antigen_id.split(";")
        df_chain_antigen = df_pdb.query("chain_id.isin(@antigen_ids)")
        if len(df_chain_antigen)== 0:
            raise ValueError(f"Empty antigen, please check pdb {pdb_code}")

        if len(df_chain_heavy)>0:
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
                labels_heavy, _, _ = get_labels(position_dict_heavy, distance_dict_heavy, alpha=alpha)
                dataset_dict[index][f"H_id labels {alpha}"] = labels_heavy
        else:
            dataset_dict[index]["H_id distances"] = []
            dataset_dict[index]["H_id numbers"] = []
            dataset_dict[index]["H_id sequence"] = ""
            dataset_dict[index]["H_id labels 4.5"] = []
            for alpha in [3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5]:
                dataset_dict[index][f"H_id labels {alpha}"] = []

        if len(df_chain_light)>0:
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


def get_other_labels_paired(
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

def get_outputs_paired(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    len_heavy: torch.Tensor,
    len_light: torch.Tensor,
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
    for i in range(len_heavy.shape[-1]):
        heavy, light = len_heavy[i], len_light[i]
        emb = get_embedding_paired(
            embedding=embedding[i],
            embedding_models=embedding_models_list,
            heavy=heavy,
            light=light,
        )
        embedding_list.append(emb)
        label_list.append(labels[i][: heavy + light])
    embedding = torch.cat(embedding_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    output = model(embedding)
    return labels, output
