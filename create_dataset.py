import json
from pathlib import Path
from typing import Dict

import ablang2
import numpy as np
import pandas as pd
import torch
import typer
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer
from utils import build_dictionary

app = typer.Typer(add_completion=False)


def create_dictionary(
    pdb_dataframe: pd.DataFrame,
    save_path : Path = Path("/home/gathenes/all_structures/data_high_qual/dataset_dict_241003.json"),
    pdbs_only: bool = False,
    pdb_folder_path:Path=Path("/home/gathenes/all_structures/imgt_renumbered_expanded"),
)-> Dict:
    """Create dictionary with indices mapping to heavy and light lists of matching imgt numbers, \
            sequences and labels.

    Args:
        pdb_dataframe (pd.DataFrame): Dataframe to use to create dataset.
        save_path (Path): Path where to save dictionary.
        pdbs_only (bool): Use only the pdbs and add all chains. Default to False.
        pdb_folder_path (Path): Path of pdb files.

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


    print("BUILDING DICTIONARY")
    dataset_dict = build_dictionary(pdbs_and_chain=pdbs_and_chain, pdb_folder_path=pdb_folder_path)
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
    """
    print("CREATING EMBEDDINGS")
    sequence_heavy_emb = [dataset_dict[index]["H_id sequence"] for index in dataset_dict]
    sequence_light_emb = [dataset_dict[index]["L_id sequence"] for index in dataset_dict]
    ########################################################
    ######################## BERT ##########################
    ########################################################
    bert_tokeniser = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
    bert_model = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False)
    paired_sequences = []
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb):
        paired_sequences.append(
            " ".join(seq_heavy) + " [SEP] " + " ".join(seq_light)
        )
    tokens = bert_tokeniser.batch_encode_plus(
        paired_sequences,
        add_special_tokens=True,
        padding="max_length",
        max_length=280,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    with torch.no_grad():
        output = bert_model(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
        )
        bert_residue_embeddings = output.last_hidden_state
    ########################################################
    ###################### IGT5 ############################
    ########################################################
    igt5_tokeniser = T5Tokenizer.from_pretrained("Exscientia/IgT5", do_lower_case=False)
    igt5_model = T5EncoderModel.from_pretrained("Exscientia/IgT5")

    tokens = igt5_tokeniser.batch_encode_plus(
        paired_sequences,
        add_special_tokens=True,
        padding="max_length",
        max_length=280,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    with torch.no_grad():
        output = igt5_model(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
        )
        igt5_residue_embeddings = output.last_hidden_state
    ########################################################
    ##################### ABLANG ###########################
    ########################################################
    ablang = ablang2.pretrained()
    all_seqs=[]
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb):
        all_seqs.append(
            [seq_heavy,seq_light]
        )
    ablang_embeddings=ablang(all_seqs, mode='rescoding', stepwise_masking = False)
    ablang_embeddings = [np.pad(each, ((0, 280-each.shape[0]),(0,0)), 'constant', constant_values=((0,0),(0,0))) for each in ablang_embeddings]
    ablang_embeddings = torch.Tensor(np.stack(ablang_embeddings))
    residue_embeddings = torch.cat([bert_residue_embeddings, igt5_residue_embeddings, ablang_embeddings], dim=2)
    torch.save(residue_embeddings, save_path)
    return residue_embeddings



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
    pdb_folder_path:Path=typer.Option(
        "/home/gathenes/all_structures/imgt_renumbered_expanded",
        "--pdb-folder-path", help="Pdb path for ground truth labeling."
    )
) -> None:
    stem=pdb_list_path.stem
    save_folder = result_folder/Path(stem)
    save_folder.mkdir(exist_ok=True,parents=True)
    pdb_dataframe = pd.read_csv(pdb_list_path)
    dataset_dict = create_dictionary(pdb_dataframe, save_path=save_folder / Path("dict.json"), pdbs_only=pdbs_only, pdb_folder_path=pdb_folder_path)
    residue_embeddings=create_embeddings(dataset_dict=dataset_dict,save_path=save_folder / Path("embeddings.pt"))

if __name__ == "__main__":
    app()
