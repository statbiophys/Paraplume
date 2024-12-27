import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict

import ablang2
import esm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import typer
from antiberty import AntiBERTyRunner
from scipy.spatial import ConvexHull
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer
from utils import build_dictionary, read_pdb_to_dataframe

app = typer.Typer(add_completion=False)
import warnings

warnings.filterwarnings('ignore')

def add_convex_hull_column(df: pd.DataFrame):
    """Add convex hull column to dataframe.

    Args:
        df (pd.DataFrame): Dataframe representing antibody.

    Returns:
        df (pd.DataFrame): Dataframe representing antibody with convex hull info per amino acid.
    """
    df_copy = deepcopy(df)
    # Initialize the convex hull column with NaN (not part of any convex hull yet)
    df_copy["convex_hull"] = np.nan
    # Extract the 3D coordinates as a numpy array
    points = df_copy[["x_coord", "y_coord", "z_coord"]].to_numpy()
    k = 1
    total_df = pd.DataFrame()
    while len(points) > 3:
        # Need at least 4 points to form a convex hull in 3D
        # Compute the convex hull for the current set of points
        hull = ConvexHull(points)
        hull_indices = hull.vertices
        # Mark these points in the DataFrame with the current iteration k
        df_copy.loc[df_copy.index[hull_indices], "convex_hull"] = k

        # Remove the points on the current convex hull
        points = np.delete(points, hull_indices, axis=0)
        total_df = pd.concat([total_df, df_copy.loc[df_copy.index[hull_indices]]])

        df_copy = df_copy.drop(df_copy.index[hull_indices])

        k += 1

    res_to_ch = total_df.set_index("IMGT")["convex_hull"].to_dict()
    df["convex_hull"] = df["IMGT"].map(res_to_ch).fillna(k).astype(int)
    return df

def create_dictionary(
    pdb_dataframe: pd.DataFrame,
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


    print("PDBS & CHAINS")
    pdbs_and_chain=pdb_dataframe.reset_index()


    print("BUILDING DICTIONARY")
    dataset_dict = build_dictionary(pdbs_and_chain=pdbs_and_chain, pdb_folder_path=pdb_folder_path)
    return dataset_dict

def create_embeddings(
        dataset_dict: Dict,
        save_path: Path = Path("/home/gathenes/paratope_model/test/test3/results"),
    ):
    """Create LLM amino acid embeddings.

    Args:
        dataset_dict (Dict): Dictionary mapping index to heavy and light aa sequence.
        save_path (Path): Path where to save embeddings.
    """
    print("CREATING EMBEDDINGS")
    sequence_heavy_emb = [dataset_dict[index]["H_id sequence"] for index in dataset_dict]
    sequence_light_emb = [dataset_dict[index]["L_id sequence"] for index in dataset_dict]
    paired_sequences = []
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb):
        paired_sequences.append(
            " ".join(seq_heavy) + " [SEP] " + " ".join(seq_light)
        )

    ########################################################
    ######################## BERT ##########################
    ########################################################
    print("CREATING IG BERT EMBEDDINGS")

    bert_tokeniser = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
    bert_model = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False)
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
    print("CREATING IG T5 EMBEDDINGS")
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
    print("CREATING ABLANG2 EMBEDDINGS")
    ablang = ablang2.pretrained()
    all_seqs = [[seq_heavy, seq_light] for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb)]
    ablang_embeddings = ablang(all_seqs, mode='rescoding', stepwise_masking=False)
    ablang_embeddings = [np.pad(each, ((0, 280 - each.shape[0]), (0, 0)), 'constant') for each in ablang_embeddings]
    ablang_embeddings = torch.Tensor(np.stack(ablang_embeddings))

    ########################################################
    ######################## ESM ###########################
    ########################################################
    print("CREATING ESM EMBEDDINGS")
    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_batch_converter = esm_alphabet.get_batch_converter()
    esm_model.eval()

    data = []
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb):
        data.append(("ab", "".join(seq_heavy) + "".join(seq_light)))
    _, _, esm_batch_tokens = esm_batch_converter(data)
    with torch.no_grad():
        esm_results = esm_model(esm_batch_tokens, repr_layers=[33], return_contacts=False)
    esm_embeddings = esm_results["representations"][33]
    pad_length = 280 - esm_embeddings.size(1)  # 280 is the desired length
    padding = (0, 0, 0, pad_length)
    esm_embeddings = F.pad(esm_embeddings, padding, mode='constant', value=0)

    ########################################################
    #################### ANTIBERTY #########################
    ########################################################
    print("CREATING ANTIBERTY EMBEDDINGS")
    antiberty = AntiBERTyRunner()
    antiberty_sequences = [
        "".join(seq_heavy) + "".join(seq_light)
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb)
    ]
    antiberty_embeddings = antiberty.embed(antiberty_sequences)
    antiberty_embeddings = [np.pad(each.cpu().numpy(), ((0, 280 - each.shape[0]), (0, 0)), 'constant') for each in antiberty_embeddings]
    antiberty_embeddings = torch.Tensor(np.stack(antiberty_embeddings))

    ########################################################
    ####################### ProtT5 #########################
    ########################################################
    print("CREATING PROT T5 EMBEDDINGS")

    device = torch.device('cpu')

    prot_t5_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    prot_t5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
    prot_t5_model.to(torch.float32)

    prot_t5_sequences = [
        "".join(seq_heavy) + "".join(seq_light)
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb)
    ]
    prot_t5_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in prot_t5_sequences]
    prot_t5_ids = prot_t5_tokenizer(prot_t5_sequences, add_special_tokens=True, padding="longest", return_tensors="pt")

    input_ids = prot_t5_ids['input_ids'].to(device)
    attention_mask = prot_t5_ids['attention_mask'].to(device)

    with torch.no_grad():
        prot_t5_output = prot_t5_model(input_ids=input_ids, attention_mask=attention_mask)

    prot_t5_embeddings = prot_t5_output.last_hidden_state
    pad_length = 280 - prot_t5_embeddings.size(1)
    padding = (0, 0, 0, pad_length)
    prot_t5_embeddings = F.pad(prot_t5_embeddings, padding, mode='constant', value=0)

    ########################################################
    ################# CONCATENATE EMBEDDINGS ###############
    ########################################################
    residue_embeddings = torch.cat([
        bert_residue_embeddings,
        igt5_residue_embeddings,
        ablang_embeddings,
        esm_embeddings,
        antiberty_embeddings,
        prot_t5_embeddings
    ], dim=2)
    torch.save(residue_embeddings, save_path)
    return residue_embeddings

def create_convex_hull(pdb_list_path:Path, pdb_folder_path:Path, dataset_dict:Dict):
    """Create convex hull value in dictionary.

    Args:
        pdb_list_path (Path): Path of csv of pdbs and their chains.
        pdb_folder_path (Path): Folder of pdb files.
        dataset_dict (Dict): Dictionary of pdbs and information.

    Returns:
        dataset_dict(Dict): Dictionary with convex hull information.
    """
    for i, value_dict in tqdm(dataset_dict.items()):
        pdb = value_dict["pdb_code"]
        Hchain, Lchain = pd.read_csv(pdb_list_path).query("pdb==@pdb")[["Hchain", "Lchain"]].values[0]
        chains = [Hchain, Lchain]
        df_pdb = (
            read_pdb_to_dataframe(f"{pdb_folder_path}/{pdb}.pdb")
            .query("chain_id.isin(@chains) and residue_number<129")
        )
        df_pdb = add_convex_hull_column(df_pdb)

        df_pdb_heavy = df_pdb.query("chain_id==@Hchain")
        ch_heavy_dict = df_pdb_heavy.set_index("IMGT")["convex_hull"].to_dict()
        heavy_numbers = value_dict["H_id numbers"]
        heavy_convex_hull = []
        for each in heavy_numbers:
            heavy_convex_hull.append(ch_heavy_dict[each])
        dataset_dict[i]["H_id convex_hull"] = heavy_convex_hull

        df_pdb_light = df_pdb.query("chain_id==@Lchain")
        df_pdb_light = df_pdb_light.query("residue_number<128")
        ch_light_dict = df_pdb_light.set_index("IMGT")["convex_hull"].to_dict()
        light_numbers = value_dict["L_id numbers"]
        light_convex_hull = []
        for each in light_numbers:
            light_convex_hull.append(ch_light_dict[each])
        dataset_dict[i]["L_id convex_hull"] = light_convex_hull
    return dataset_dict

def create_convex_hull_mean(pdb_list_path:Path, pdb_folder_path:Path, dataset_dict:Dict):
    """Create mean convex hull value in dictionary.

    Args:
        pdb_list_path (Path): Path of csv of pdbs and their chains.
        pdb_folder_path (Path): Folder of pdb files.
        dataset_dict (Dict): Dictionary of pdbs and information.

    Returns:
        dataset_dict(Dict): Dictionary with mean convex hull information.
    """
    for i, value_dict in tqdm(dataset_dict.items()):
        pdb = value_dict["pdb_code"]
        Hchain, Lchain = pd.read_csv(pdb_list_path).query("pdb==@pdb")[["Hchain", "Lchain"]].values[0]
        chains = [Hchain, Lchain]
        df_pdb = (
            read_pdb_to_dataframe(f"{pdb_folder_path}/{pdb}.pdb")
            .query("chain_id.isin(@chains) and residue_number<129")
        )
        df_pdb = add_convex_hull_column(df_pdb)

        df_pdb_heavy = df_pdb.query("chain_id==@Hchain")
        df_pdb_heavy = df_pdb_heavy.groupby("IMGT", as_index=False)["convex_hull"].mean()
        ch_heavy_dict = df_pdb_heavy.set_index("IMGT")["convex_hull"].to_dict()
        heavy_numbers = value_dict["H_id numbers"]
        heavy_convex_hull = []
        for each in heavy_numbers:
            heavy_convex_hull.append(ch_heavy_dict[each])
        dataset_dict[i]["H_id convex_hull mean"] = heavy_convex_hull

        df_pdb_light = df_pdb.query("chain_id==@Lchain and residue_number<128")
        df_pdb_light = df_pdb_light.groupby("IMGT", as_index=False)["convex_hull"].mean()
        ch_light_dict = df_pdb_light.set_index("IMGT")["convex_hull"].to_dict()
        light_numbers = value_dict["L_id numbers"]
        light_convex_hull = []
        for each in light_numbers:
            light_convex_hull.append(ch_light_dict[each])
        dataset_dict[i]["L_id convex_hull mean"] = light_convex_hull
    return dataset_dict

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
    pdb_folder_path:Path=typer.Option(
        "/home/gathenes/all_structures/imgt_renumbered_expanded",
        "--pdb-folder-path", help="Pdb path for ground truth labeling."
    )
) -> None:
    stem=pdb_list_path.stem
    save_folder = result_folder/Path(stem)
    save_folder.mkdir(exist_ok=True,parents=True)
    pdb_dataframe = pd.read_csv(pdb_list_path)
    dataset_dict = create_dictionary(pdb_dataframe, pdb_folder_path=pdb_folder_path)
    residue_embeddings = create_embeddings(dataset_dict=dataset_dict,save_path=save_folder / Path("embeddings.pt"))
    dataset_dict = create_convex_hull(pdb_list_path=pdb_list_path,pdb_folder_path=pdb_folder_path,dataset_dict=dataset_dict)
    dataset_dict = create_convex_hull_mean(pdb_list_path=pdb_list_path,pdb_folder_path=pdb_folder_path,dataset_dict=dataset_dict)
    with open(save_folder / Path("dict.json"), "w") as f:
        json.dump(dataset_dict, f)



if __name__ == "__main__":
    app()
