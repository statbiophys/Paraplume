"""Create dictionary of sequence/labels and corresponding LLM embeddings."""

import json
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List

import ablang2
import esm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import typer
from antiberty import AntiBERTyRunner
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer

from paraplume.utils import get_logger
from paraplume.utils_paired import build_dictionary_paired

app = typer.Typer(add_completion=False)
log = get_logger()
warnings.filterwarnings("ignore")
# pylint: disable=E1101


def create_embeddings_from_dict(
    dataset_dict: Dict[str, Dict[str, Any]],
    save_path: Path,
    emb_proc_size: int = 100,
    heavy_chain:bool=False,
    gpu:int=0,
) -> torch.Tensor:
    """Create LLM amino acid embeddings.

    Args:
        dataset_dict (Dict[str, Dict[str, Any]]): Dictionary mapping index to heavy and light aa \
            sequence.
        save_path (Path): Path where to save embeddings.
        emb_proc_size (int): Batch size to create embeddings without memory explosion.


    Returns:
        torch.Tensor: Tensor of embeddings.
    """
    log.info("CREATING EMBEDDINGS")
    sequence_heavy_emb = [dataset_dict[index]["H_id sequence"] for index in dataset_dict]
    sequence_light_emb = [dataset_dict[index]["L_id sequence"] for index in dataset_dict]

    residue_embeddings = create_emebddings_from_seq(sequence_heavy_emb, sequence_light_emb,emb_proc_size, heavy_chain=heavy_chain, gpu=gpu)
    torch.save(residue_embeddings, save_path)
    return residue_embeddings

def create_emebddings_from_seq(sequence_heavy_emb:List, sequence_light_emb:List,emb_proc_size:int, heavy_chain:bool=False, gpu:int=0):
    """

    Args:
        sequence_heavy_emb (List): Heavy sequences
        sequence_light_emb (List): Light sequences
        emb_proc_size (int): Batch size to create embeddings without memory explosion.

    Returns:
        torch.Tensor: Embeddings
    """
    log.info("CREATING EMBEDDINGS", embedding_model="IgBert")
    bert_residue_embeddings = process_batch(
        compute_igbert_embeddings, sequence_heavy_emb, sequence_light_emb, emb_proc_size, heavy_chain=heavy_chain, gpu=gpu,
    )
    log.info("CREATING EMBEDDINGS", embedding_model="IgT5")
    igt5_residue_embeddings = process_batch(
        compute_igt5_embeddings, sequence_heavy_emb, sequence_light_emb, emb_proc_size, heavy_chain=heavy_chain, gpu=gpu,
    )
    log.info("CREATING EMBEDDINGS", embedding_model="Ablang2")
    ablang_embeddings = process_batch(
        compute_ablang_embeddings, sequence_heavy_emb, sequence_light_emb, emb_proc_size, gpu=gpu,
    )
    log.info("CREATING EMBEDDINGS", embedding_model="ESM")
    esm_embeddings = process_batch(
        compute_esm_embeddings, sequence_heavy_emb, sequence_light_emb, emb_proc_size, gpu=gpu,
    )
    log.info("CREATING EMBEDDINGS", embedding_model="Antiberty")
    antiberty_embeddings = process_batch(
        compute_antiberty_embeddings,
        sequence_heavy_emb,
        sequence_light_emb,
        emb_proc_size,
        gpu=gpu,
    )
    log.info("CREATING EMBEDDINGS", embedding_model="Prot-T5")
    prot_t5_embeddings = process_batch(
        compute_t5_embeddings, sequence_heavy_emb, sequence_light_emb, emb_proc_size, gpu=gpu,
    )
    ########################################################
    ################# CONCATENATE EMBEDDINGS ###############
    ########################################################
    residue_embeddings = torch.cat(
        [
            bert_residue_embeddings,
            igt5_residue_embeddings,
            ablang_embeddings,
            esm_embeddings,
            antiberty_embeddings,
            prot_t5_embeddings,
        ],
        dim=2,
    )
    return residue_embeddings

def get_llm_to_embedding_dict(
    sequence_heavy:List,
    sequence_light:List,
    emb_proc_size:int,
    llm_list:List[str],
    heavy_chain:bool=False,
    gpu:int=0):
    """

    Args:
        sequence_heavy_emb (List): Heavy sequences
        sequence_light_emb (List): Light sequences
        emb_proc_size (int): Batch size to create embeddings without memory explosion.

    Returns:
        torch.Tensor: Embeddings
    """
    llm_to_func =  {
        "ablang2": compute_ablang_embeddings,
        "igT5": compute_igt5_embeddings,
        "igbert": compute_igbert_embeddings,
        "esm": compute_esm_embeddings,
        "antiberty": compute_antiberty_embeddings,
        "prot-t5": compute_t5_embeddings,
    }
    llm_to_emb={}
    for llm in llm_list:
        log.info("CREATING EMBEDDINGS", embedding_model=llm)
        embedding = process_batch(llm_to_func[llm],sequence_heavy, sequence_light, emb_proc_size, heavy_chain=heavy_chain, gpu=gpu)
        llm_to_emb[llm]=embedding
    return llm_to_emb

def process_batch(
    func: Callable, heavy_sequences: List, light_sequences: List, emb_proc_size: int, heavy_chain:bool=False,gpu:int=0,
) -> torch.Tensor:
    """Create embedding for batch of size emb_proc_size.

    Args:
        func (Callable): Embedding function to use.
        heavy_sequences (List): List of heavy sequences.
        light_sequences (List): List of light sequences.
        emb_proc_size (int): Size of the batch.

    Returns:
        torch.Tensor: Tensor of embedding.
    """
    batch_embeddings = []
    for i in tqdm(range(0, max(len(heavy_sequences), len(light_sequences)), emb_proc_size)):
        heavy_batch = heavy_sequences[i : i + emb_proc_size]
        light_batch = light_sequences[i : i + emb_proc_size]
        batch_embeddings.append(func(heavy_batch, light_batch, heavy_chain=heavy_chain,gpu=gpu))
    concat_embeddings = torch.cat(batch_embeddings, dim=0)
    return concat_embeddings


def compute_antiberty_embeddings(
    sequence_heavy_emb: List, sequence_light_emb: List, heavy_chain:bool=False, gpu:int=0
) -> torch.Tensor:
    """Compute embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.

    Returns:
        torch.Tensor: Antiberty embeddings.
    """
    antiberty = AntiBERTyRunner()  # Move to GPU
    antiberty_sequences = [
        "".join(seq_heavy) + "".join(seq_light)
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb)
    ]
    antiberty_embeddings: List[torch.Tensor] = antiberty.embed(antiberty_sequences)
    antiberty_embeddings = [
        np.pad(
            each.cpu().numpy(),  # Move to CPU before padding
            ((0, 285 - each.shape[0]), (0, 0)),
            "constant",
        )
        for each in antiberty_embeddings
    ]
    antiberty_embeddings = torch.Tensor(np.stack(antiberty_embeddings))  # Convert to tensor
    return antiberty_embeddings


def compute_esm_embeddings(sequence_heavy_emb: List, sequence_light_emb: List, heavy_chain:bool=False, gpu:int=0) -> torch.Tensor:
    """Compute embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.

    Returns:
        torch.Tensor: ESM embeddings.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)  # Move model to GPU
    esm_batch_converter = esm_alphabet.get_batch_converter()
    esm_model.eval()
    valid_characters = set(esm_alphabet.all_toks)

    data = []
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb):
        cleaned_seq_heavy = "".join([char if char in valid_characters else "X" for char in seq_heavy])
        cleaned_seq_light = "".join([char if char in valid_characters else "X" for char in seq_light])
        data.append(("ab", "".join(cleaned_seq_heavy) + "".join(cleaned_seq_light)))
    _, _, esm_batch_tokens = esm_batch_converter(data)
    esm_batch_tokens = esm_batch_tokens.to(device)  # Move to GPU
    with torch.no_grad():
        esm_results = esm_model(esm_batch_tokens, repr_layers=[33], return_contacts=False)
    esm_embeddings = esm_results["representations"][33]
    pad_length = 285 - esm_embeddings.size(1)  # 285 is the desired length
    padding = (0, 0, 0, pad_length)
    esm_embeddings = F.pad(esm_embeddings, padding, mode="constant", value=0)
    return esm_embeddings.cpu()  # Move back to CPU


def compute_ablang_embeddings(sequence_heavy_emb: List, sequence_light_emb: List, heavy_chain:bool=False, gpu:int=0) -> torch.Tensor:
    """Compute embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.

    Returns:
        torch.Tensor: Ablang2 embeddings.
    """
    ablang = ablang2.pretrained()  # Move to GPU
    all_seqs = [
        [seq_heavy, seq_light]
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb)
    ]
    ablang_embeddings = ablang(all_seqs, mode="rescoding", stepwise_masking=False)
    ablang_embeddings = [
        np.pad(each, ((0, 285 - each.shape[0]), (0, 0)), "constant") for each in ablang_embeddings
    ]
    ablang_embeddings = torch.Tensor(np.stack(ablang_embeddings))  # Ensure it's on GPU
    return ablang_embeddings


def compute_igt5_embeddings(sequence_heavy_emb: List, sequence_light_emb: List, heavy_chain:bool=False, gpu:int=0) -> torch.Tensor:
    """Compute embeddings.

    Args:
        paired_sequences (List): Paired sequences of heavy and light chains.

    Returns:
        torch.Tensor: IgT5 embeddings.
    """
    if heavy_chain:
        sequence_emb=sequence_heavy_emb
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        sequences=[" ".join(seq) for seq in sequence_emb]
        igt5_tokeniser = T5Tokenizer.from_pretrained("Exscientia/IgT5_unpaired", do_lower_case=False)
        igt5_model = T5EncoderModel.from_pretrained("Exscientia/IgT5_unpaired").to(device)  # Move to GPU
        tokens = igt5_tokeniser.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding="max_length",
            max_length=285,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}  # Move tokens to GPU
        with torch.no_grad():
            output = igt5_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
            igt5_residue_embeddings = output.last_hidden_state
        return igt5_residue_embeddings.cpu()  # Move back to CPU
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    paired_sequences = []
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb):
        paired_sequences.append(" ".join(seq_heavy) + " </s> " + " ".join(seq_light))
    igt5_tokeniser = T5Tokenizer.from_pretrained("Exscientia/IgT5", do_lower_case=False)
    igt5_model = T5EncoderModel.from_pretrained("Exscientia/IgT5").to(device)  # Move to GPU
    tokens = igt5_tokeniser.batch_encode_plus(
        paired_sequences,
        add_special_tokens=True,
        padding="max_length",
        max_length=285,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}  # Move tokens to GPU
    with torch.no_grad():
        output = igt5_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        igt5_residue_embeddings = output.last_hidden_state
    return igt5_residue_embeddings.cpu()  # Move back to CPU


def compute_igbert_embeddings(sequence_heavy_emb: List, sequence_light_emb: List, heavy_chain:bool=False, gpu:int=0) -> torch.Tensor:
    """Compute embeddings.

    Args:
        paired_sequences (List): Paired sequences of heavy and light chains.

    Returns:
        torch.Tensor: Ig BERT embeddings.
    """
    if heavy_chain:
        sequence_emb=sequence_heavy_emb
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        sequences=[" ".join(seq) for seq in sequence_emb]
        bert_tokeniser = BertTokenizer.from_pretrained("Exscientia/IgBert_unpaired", do_lower_case=False)
        bert_model = BertModel.from_pretrained("Exscientia/IgBert_unpaired", add_pooling_layer=False).to(device)  # Move to GPU
        tokens = bert_tokeniser.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding="max_length",
            max_length=285,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}  # Move tokens to GPU
        with torch.no_grad():
            output = bert_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
            bert_residue_embeddings = output.last_hidden_state
        return bert_residue_embeddings.cpu()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    paired_sequences = []
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb):
        paired_sequences.append(" ".join(seq_heavy) + " [SEP] " + " ".join(seq_light))
    bert_tokeniser = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
    bert_model = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False).to(device)  # Move to GPU
    tokens = bert_tokeniser.batch_encode_plus(
        paired_sequences,
        add_special_tokens=True,
        padding="max_length",
        max_length=285,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}  # Move tokens to GPU
    with torch.no_grad():
        output = bert_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        bert_residue_embeddings = output.last_hidden_state
    return bert_residue_embeddings.cpu()  # Move back to CPU


def compute_t5_embeddings(sequence_heavy_emb: List, sequence_light_emb: List, heavy_chain:bool=False, gpu:int=0) -> torch.Tensor:
    """Compute embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.

    Returns:
        torch.Tensor: Prot-T5 embeddings.
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    prot_t5_tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    prot_t5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)  # Move to GPU

    prot_t5_sequences = [
        "".join(seq_heavy) + "".join(seq_light)
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb)
    ]
    prot_t5_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in prot_t5_sequences]
    prot_t5_ids = prot_t5_tokenizer(
        prot_t5_sequences,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    prot_t5_ids = {key: value.to(device) for key, value in prot_t5_ids.items()}  # Move tokens to GPU

    with torch.no_grad():
        prot_t5_output = prot_t5_model(input_ids=prot_t5_ids["input_ids"], attention_mask=prot_t5_ids["attention_mask"])

    prot_t5_embeddings = prot_t5_output.last_hidden_state
    pad_length = 285 - prot_t5_embeddings.size(1)
    padding = (0, 0, 0, pad_length)
    prot_t5_embeddings = F.pad(prot_t5_embeddings, padding, mode="constant", value=0)
    return prot_t5_embeddings.cpu()  # Move back to CPU

@app.command()
def main(
    csv_file_path: Path = typer.Argument(
        ...,
        help="Path of csv file to use for pdb list.",
        show_default=False,
    ),
    result_folder: Path = typer.Option(
        Path("./result/"), "--result-folder", "-r", help="Where to save results"
    ),
    pdb_folder_path: Path = typer.Option(
        "/home/athenes/all_structures/imgt_renumbered_expanded",
        "--pdb-folder-path",
        help="Pdb path for ground truth labeling.",
    ),
    emb_proc_size: int = typer.Option(
        100,
        "--emb-proc-size",
        help="We create embeddings batch by batch to avoid memory explosion. This is the batch\
            size. Optimal value depends on your computer. Defaults to 100.",
    ),
    heavy_chain:bool=typer.Option(
        False,
        "--heavy-chain",
        help="Chain to use for single chain model. Defaults to heavy.",
    ),
    gpu:int=typer.Option(
        0,
        "--gpu",
        help="Which gpu to use",
    )
) -> None:
    """Create and save dictionary of sequences and labels in a json file, and the corresponding  \
        LLM embeddings in  pt file."""
    stem = csv_file_path.stem
    save_folder = result_folder / Path(stem)
    save_folder.mkdir(exist_ok=True, parents=True)
    pdb_dataframe = pd.read_csv(csv_file_path).reset_index()
    dataset_dict = build_dictionary_paired(pdb_dataframe=pdb_dataframe, pdb_folder_path=pdb_folder_path)
    _ = create_embeddings_from_dict(
        dataset_dict=dataset_dict,
        save_path=save_folder / Path("embeddings.pt"),
        emb_proc_size=emb_proc_size,
        heavy_chain=heavy_chain,
        gpu=gpu,
    )
    with open(save_folder / Path("dict.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f)


if __name__ == "__main__":
    app()
