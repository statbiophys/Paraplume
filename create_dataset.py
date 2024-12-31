"""Create dictionary of sequence/labels and corresponding LLM embeddings."""
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import ablang2
import esm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import typer
from antiberty import AntiBERTyRunner
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer

from utils import build_dictionary, get_logger

app = typer.Typer(add_completion=False)
log = get_logger()
warnings.filterwarnings("ignore")
# pylint: disable=E1101


def create_embeddings(
    dataset_dict: Dict[str, Dict[str, Any]],
    save_path: Path,
) -> torch.Tensor:
    """Create LLM amino acid embeddings.

    Args:
        dataset_dict (Dict[str, Dict[str, Any]]): Dictionary mapping index to heavy and light aa \
            sequence.
        save_path (Path): Path where to save embeddings.

    Returns:
        torch.Tensor: Tensor of embeddings.
    """
    log.info("CREATING EMBEDDINGS")
    sequence_heavy_emb = [dataset_dict[index]["H_id sequence"] for index in dataset_dict]
    sequence_light_emb = [dataset_dict[index]["L_id sequence"] for index in dataset_dict]
    paired_sequences = []
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb):
        paired_sequences.append(" ".join(seq_heavy) + " [SEP] " + " ".join(seq_light))

    log.info("CREATING EMBEDDINGS", embedding_model="IgBert")
    bert_residue_embeddings = compute_igbert_embeddings(paired_sequences)
    log.info("CREATING EMBEDDINGS", embedding_model="IgT5")
    igt5_residue_embeddings = compute_igt5_embeddings(paired_sequences)
    log.info("CREATING EMBEDDINGS", embedding_model="Ablang2")
    ablang_embeddings = compute_ablang_embeddings(sequence_heavy_emb, sequence_light_emb)
    log.info("CREATING EMBEDDINGS", embedding_model="ESM")
    esm_embeddings = compute_esm_embeddings(sequence_heavy_emb, sequence_light_emb)
    log.info("CREATING EMBEDDINGS", embedding_model="Antiberty")
    antiberty_embeddings = compute_antiberty_embeddings(sequence_heavy_emb, sequence_light_emb)
    log.info("CREATING EMBEDDINGS", embedding_model="Prot-T5")
    prot_t5_embeddings = compute_t5_embeddings(sequence_heavy_emb, sequence_light_emb)

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
    torch.save(residue_embeddings, save_path)
    return residue_embeddings


def compute_antiberty_embeddings(
    sequence_heavy_emb: List, sequence_light_emb: List
) -> torch.Tensor:
    """Compute embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.

    Returns:
        torch.Tensor: Antiberty embeddings.
    """
    antiberty = AntiBERTyRunner()
    antiberty_sequences = [
        "".join(seq_heavy) + "".join(seq_light)
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb)
    ]
    antiberty_embeddings: List[torch.Tensor] = antiberty.embed(antiberty_sequences)
    antiberty_embeddings = [
        np.pad(
            each.cpu().numpy(),
            ((0, 280 - each.shape[0]), (0, 0)),
            "constant",
        )
        for each in antiberty_embeddings
    ]
    antiberty_embeddings = torch.Tensor(np.stack(antiberty_embeddings))
    return antiberty_embeddings


def compute_esm_embeddings(sequence_heavy_emb: List, sequence_light_emb: List) -> torch.Tensor:
    """Compute embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.

    Returns:
        torch.Tensor: ESM embeddings.
    """
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
    esm_embeddings = F.pad(esm_embeddings, padding, mode="constant", value=0)
    return esm_embeddings


def compute_ablang_embeddings(sequence_heavy_emb: List, sequence_light_emb: List) -> torch.Tensor:
    """Compute embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.

    Returns:
        torch.Tensor: Ablang2 embeddings.
    """
    ablang = ablang2.pretrained()
    all_seqs = [
        [seq_heavy, seq_light]
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb)
    ]
    ablang_embeddings = ablang(all_seqs, mode="rescoding", stepwise_masking=False)
    ablang_embeddings = [
        np.pad(each, ((0, 280 - each.shape[0]), (0, 0)), "constant") for each in ablang_embeddings
    ]
    ablang_embeddings = torch.Tensor(np.stack(ablang_embeddings))
    return ablang_embeddings


def compute_igt5_embeddings(paired_sequences: List) -> torch.Tensor:
    """Compute embeddings.

    Args:
        paired_sequences (List): Paired sequences of heavy and light chains.

    Returns:
        torch.Tensor: IgT5 embeddings.
    """
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
        output = igt5_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        igt5_residue_embeddings = output.last_hidden_state
    return igt5_residue_embeddings


def compute_igbert_embeddings(paired_sequences: List) -> torch.Tensor:
    """Compute embeddings.

    Args:
        paired_sequences (List): Paired sequences of heavy and light chains.

    Returns:
        torch.Tensor: Ig bert embeddings.
    """
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
        output = bert_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        bert_residue_embeddings = output.last_hidden_state
    return bert_residue_embeddings


def compute_t5_embeddings(sequence_heavy_emb: List, sequence_light_emb: List) -> torch.Tensor:
    """Compute embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.

    Returns:
        torch.Tensor: Prot-T5 embeddings.
    """
    device = torch.device("cpu")

    prot_t5_tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    prot_t5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(
        device
    )
    prot_t5_model.to(torch.float32)

    prot_t5_sequences = [
        "".join(seq_heavy) + "".join(seq_light)
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb)
    ]
    prot_t5_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in prot_t5_sequences]
    prot_t5_ids = prot_t5_tokenizer(
        prot_t5_sequences, add_special_tokens=True, padding="longest", return_tensors="pt"
    )

    input_ids = prot_t5_ids["input_ids"].to(device)
    attention_mask = prot_t5_ids["attention_mask"].to(device)

    with torch.no_grad():
        prot_t5_output = prot_t5_model(input_ids=input_ids, attention_mask=attention_mask)

    prot_t5_embeddings = prot_t5_output.last_hidden_state
    pad_length = 280 - prot_t5_embeddings.size(1)
    padding = (0, 0, 0, pad_length)
    prot_t5_embeddings = F.pad(prot_t5_embeddings, padding, mode="constant", value=0)
    return prot_t5_embeddings


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
) -> None:
    """Create and save dictionary of sequences and labels in a json file, and the corresponding  \
        LLM embeddings in  pt file."""
    stem = csv_file_path.stem
    save_folder = result_folder / Path(stem)
    save_folder.mkdir(exist_ok=True, parents=True)
    pdb_dataframe = pd.read_csv(csv_file_path).reset_index()
    dataset_dict = build_dictionary(pdb_dataframe=pdb_dataframe, pdb_folder_path=pdb_folder_path)
    _ = create_embeddings(dataset_dict=dataset_dict, save_path=save_folder / Path("embeddings.pt"))
    with open(save_folder / Path("dict.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f)


if __name__ == "__main__":
    app()
