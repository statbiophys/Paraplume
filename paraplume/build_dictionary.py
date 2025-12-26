"""Create dictionary of sequence/labels and corresponding LLM embeddings."""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from paraplume.utils import get_binding_residues, get_labels, get_logger, read_pdb_to_dataframe

app = typer.Typer(add_completion=False)
log = get_logger()
warnings.filterwarnings("ignore")


def rec_dd():
    """Return nested default_dict.

    Returns
    -------
        defaultdict: Dictionary where you can add as many nested keys as you wish.
    """
    return defaultdict(rec_dd)


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
        else:
            dataset_dict[index]["H_id distances"] = []
            dataset_dict[index]["H_id numbers"] = []
            dataset_dict[index]["H_id sequence"] = ""
            dataset_dict[index]["H_id labels 4.5"] = []

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

        else:
            dataset_dict[index]["L_id distances"] = []
            dataset_dict[index]["L_id numbers"] = []
            dataset_dict[index]["L_id sequence"] = ""
            dataset_dict[index]["L_id labels 4.5"] = []

        dataset_dict[index]["pdb_code"] = pdb_code

    return dataset_dict


@app.command()
def main(
    csv_file_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path of csv file to use for pdb list.",
        show_default=False,
    ),
    pdb_folder_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Pdb path for ground truth labeling.",
        show_default=False,
    ),
    result_folder: Path = typer.Option(  # noqa: B008
        Path("./result/"), "--result-folder", "-r", help="Where to save results."
    ),
) -> None:
    """Create dataset to train the neural network.

    Sequences and labels are saved in a .json file, and LPLM embeddings are saved in a .pt file.
    """
    stem = csv_file_path.stem
    save_folder = result_folder / Path(stem)
    save_folder.mkdir(exist_ok=True, parents=True)
    pdb_dataframe = pd.read_csv(csv_file_path).reset_index()
    dataset_dict = build_dictionary(pdb_dataframe=pdb_dataframe, pdb_folder_path=pdb_folder_path)
    with (save_folder / Path("dict.json")).open("w", encoding="utf-8") as f:
        json.dump(dataset_dict, f)


if __name__ == "__main__":
    app()
