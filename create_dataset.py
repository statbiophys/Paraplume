import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import typer
from torch_dataset import ParatopeDataset
from transformers import BertModel, BertTokenizer
from utils import build_dictionary, remove_ids

app = typer.Typer(add_completion=False)

def rec_dd():
    return defaultdict(rec_dd)

def create_dictionary(
    pdb_dataframe: pd.DataFrame,
    save_path : Path = Path("/home/gathenes/all_structures/data_high_qual/dataset_dict_241003.json"),
    pdbs_only: bool = False,
    remove_flag:bool=False,
)-> Dict:
    """Create dictionary with indices mapping to heavy and light lists of matching imgt numbers, \
            sequences and labels.

    Args:
        pdb_dataframe (pd.DataFrame): Dataframe to use to create dataset.
        save_path (Path): Path where to save dictionary.
        pdbs_only (bool): Use only the pdbs and add all chains. Default to False.

    Returns:
        (Dict) : Dictionary with indices mapping to heavy and light lists of matching imgt numbers,\
            sequences and labels.
    """

    if pdbs_only:
        print("PDBS ONLY")
        data = pd.read_csv(Path("/home/gathenes/all_structures/sabdab_summary_all.tsv"), sep="\t")
        pdb_list = pdb_dataframe["pdb"].unique()
        pdbs_and_chain = data.query("pdb in @pdb_list").dropna(
            subset=["Hchain", "Lchain", "antigen_chain"]
        )[["pdb", "Hchain", "Lchain", "antigen_chain"]]
        pdbs_and_chain.reset_index(inplace=True)
    else :
        print("PDBS & CHAINS")
        pdbs_and_chain=pdb_dataframe.reset_index()

    if remove_flag:
        print("REMOVING IDS")
        pdbs_and_chain=remove_ids(pdbs_and_chain=pdbs_and_chain)
    print("BUILDING DICTIONARY")
    dataset_dict = build_dictionary(pdbs_and_chain=pdbs_and_chain)
    with open(save_path, "w") as f:
        json.dump(dataset_dict, f)
    with open(save_path) as f:
        dataset_dict = json.load(f)
    return dataset_dict

def create_embeddings(
        dataset_dict:Dict,
        save_path : Path = Path('/home/gathenes/all_structures/data_high_qual/embeddings_241003.pt'),
    ):
    """Create LLM amino acid embeddings.

    Args:
        dataset_dict_path (Dict): Dictionary mapping index to heavy and light aa sequence.
        save_path (Path): Path where to save embeddings.
        max_length (int): Max length of your paired sequences.
    """
    print("CREATING EMBEDDINGS")
    sequence_heavy_emb = [dataset_dict[index]["H_id sequence"] for index in dataset_dict]
    sequence_light_emb = [dataset_dict[index]["L_id sequence"] for index in dataset_dict]
    tokeniser = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
    model = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False)
    paired_sequences = []
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb):
        paired_sequences.append(
            " ".join(seq_heavy) + " [SEP] " + " ".join(seq_light)
        )
    tokens = tokeniser.batch_encode_plus(
        paired_sequences,
        add_special_tokens=True,
        padding="max_length",
        max_length=256+3,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    with torch.no_grad():
        output = model(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
        )
        residue_embeddings = output.last_hidden_state
    torch.save(residue_embeddings, save_path)
    return residue_embeddings

def create_dataloader(dataset_dict:Dict,residue_embeddings:torch.Tensor, batch_size=10, shuffle:bool=False, alpha:str="4.5")->torch.utils.data.dataloader.DataLoader:
    """Take dataset_dict and embeddings and return dataloader.

    Args:
        dataset_dict (Dict): _description_
        residue_embeddings (torch.Tensor): _description_
        batch_size (int, optional): _description_. Defaults to 10.

    Returns:
        torch.utils.data.dataloader.DataLoader: Dataloader to use for training.
    """
    dataset = ParatopeDataset(dataset_dict=dataset_dict, residue_embeddings=residue_embeddings, alpha=alpha)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader

@app.command()
def main(
    pdb_list_path: Path = typer.Argument(
        ...,
        help="Path of csv file to use for pdb list.",
        show_default=False,
    ),
    result_folder: Path = typer.Option(
        Path("./result/"), "--result-folder", "-r", help="Where to save results"
    ),
    pdbs_only:bool = typer.Option(
        False, "--pdbs-only", help="Use only pdbs."
    ),
    remove_flag:bool=typer.Option(
        False, "--remove", help="Whether or not to remove problematic pdbs. Defaults to False."
    )
) -> None:
    stem=pdb_list_path.stem
    save_folder = result_folder/Path(stem)
    save_folder.mkdir(exist_ok=True,parents=True)
    pdb_dataframe = pd.read_csv(pdb_list_path)
    dataset_dict = create_dictionary(pdb_dataframe, save_path=save_folder / Path("dict.json"), pdbs_only=pdbs_only, remove_flag=remove_flag)
    residue_embeddings=create_embeddings(dataset_dict=dataset_dict,save_path=save_folder / Path("embeddings.pt"))

if __name__ == "__main__":
    app()
