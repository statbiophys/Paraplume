"""Create dictionary of sequence/labels and corresponding LLM embeddings."""

import json
import re
import warnings
from collections.abc import Callable
from pathlib import Path

import ablang2
import esm
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import typer
from antiberty import AntiBERTyRunner
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer

from paraplume.utils import get_device, get_logger

app = typer.Typer(add_completion=False)
log = get_logger()
warnings.filterwarnings("ignore")


def process_batch(  # noqa: PLR0913
    func: Callable,
    heavy_sequences: list,
    light_sequences: list,
    emb_proc_size: int,
    gpu: int = 0,
    *,
    single_chain: bool = False,
) -> torch.Tensor:
    """Create embedding for batch of size emb_proc_size for the sequences.

    Args:
        func (Callable): Embedding function to use.
        heavy_sequences (List): List of heavy sequences.
        light_sequences (List): List of light sequences.
        emb_proc_size (int): Size of the batch.
        gpu (int): Gpu to use.
        single_chain (bool): Single chain mode.

    Returns
    -------
        torch.Tensor: Tensor of embedding.
    """
    batch_embeddings = []
    for i in tqdm(range(0, max(len(heavy_sequences), len(light_sequences)), emb_proc_size)):
        heavy_batch = heavy_sequences[i : i + emb_proc_size]
        light_batch = light_sequences[i : i + emb_proc_size]
        batch_embeddings.append(func(heavy_batch, light_batch, single_chain=single_chain, gpu=gpu))
    return torch.cat(batch_embeddings, dim=0)


def compute_antiberty_embeddings(
    sequence_heavy_emb: list,
    sequence_light_emb: list,
    gpu: int = 0,  # noqa: ARG001
    *,
    single_chain: bool = False,  # noqa: ARG001
) -> torch.Tensor:
    """Compute antiberty embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.
        gpu (int): Gpu to use.
        single_chain (bool): Single chain mode.

    Returns
    -------
        torch.Tensor: Antiberty embeddings.
    """
    antiberty = AntiBERTyRunner()  # Move to GPU
    antiberty_sequences = [
        "".join(seq_heavy) + "".join(seq_light)
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb, strict=False)
    ]
    antiberty_embeddings: list[torch.Tensor] = antiberty.embed(antiberty_sequences)
    antiberty_embeddings = [each[1:, :] for each in antiberty_embeddings]
    antiberty_embeddings = [
        np.pad(
            each.cpu().numpy(),  # Move to CPU before padding
            ((0, 285 - each.shape[0]), (0, 0)),
            "constant",
        )
        for each in antiberty_embeddings
    ]
    return torch.Tensor(np.stack(antiberty_embeddings))


def compute_ablang_embeddings(
    sequence_heavy_emb: list,
    sequence_light_emb: list,
    gpu: int = 0,  # noqa: ARG001
    *,
    single_chain: bool = False,  # noqa: ARG001
) -> torch.Tensor:
    """Compute ablang-2 embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.
        gpu (int): Gpu to use.
        single_chain (bool): Single chain mode.

    Returns
    -------
        torch.Tensor: Ablang2 embeddings.
    """
    ablang = ablang2.pretrained()  # Move to GPU
    all_seqs = [
        [seq_heavy, seq_light]
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb, strict=False)
    ]
    ablang_embeddings = ablang(all_seqs, mode="rescoding", stepwise_masking=False)
    lenghts_heavy = [len(seq_heavy) for seq_heavy in sequence_heavy_emb]
    ablang_embeddings = [
        np.concatenate([each[1 : len_heavy + 1, :], each[len_heavy + 4 :, :]], axis=0)
        for each, len_heavy in zip(ablang_embeddings, lenghts_heavy, strict=False)
    ]
    ablang_embeddings = [
        np.pad(each, ((0, 285 - each.shape[0]), (0, 0)), "constant") for each in ablang_embeddings
    ]
    return torch.Tensor(np.stack(ablang_embeddings))


def compute_esm_embeddings(
    sequence_heavy_emb: list,
    sequence_light_emb: list,
    gpu: int = 0,
    *,
    single_chain: bool = False,  # noqa: ARG001
) -> torch.Tensor:
    """Compute esm embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.
        gpu (int): Gpu to use.
        single_chain (bool): Single chain mode.

    Returns
    -------
        torch.Tensor: ESM embeddings.
    """
    device = get_device(gpu)

    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)  # Move model to GPU
    esm_batch_converter = esm_alphabet.get_batch_converter()
    esm_model.eval()
    valid_characters = set(esm_alphabet.all_toks)

    data = []
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb, strict=False):
        cleaned_seq_heavy = "".join(
            [char if char in valid_characters else "X" for char in seq_heavy]
        )
        cleaned_seq_light = "".join(
            [char if char in valid_characters else "X" for char in seq_light]
        )
        data.append(("ab", "".join(cleaned_seq_heavy) + "".join(cleaned_seq_light)))
    _, _, esm_batch_tokens = esm_batch_converter(data)
    esm_batch_tokens = esm_batch_tokens.to(device)  # Move to GPU
    with torch.no_grad():
        esm_results = esm_model(esm_batch_tokens, repr_layers=[33], return_contacts=False)
    esm_embeddings = esm_results["representations"][33]
    esm_embeddings = esm_embeddings[:, 1:, :]
    pad_length = 285 - esm_embeddings.size(1)  # 285 is the desired length
    padding = (0, 0, 0, pad_length)
    return F.pad(esm_embeddings, padding, mode="constant", value=0).cpu()


def compute_igbert_embeddings(
    sequence_heavy_emb: list,
    sequence_light_emb: list,
    gpu: int = 0,
    *,
    single_chain: bool = False,
) -> torch.Tensor:
    """Compute igbert embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.
        gpu (int): Gpu to use.
        single_chain (bool): Single chain mode.

    Returns
    -------
        torch.Tensor: Ig BERT embeddings.
    """
    if single_chain:
        sequence_emb = sequence_heavy_emb if len(sequence_heavy_emb) > 0 else sequence_light_emb
        device = get_device(gpu)
        sequences = [" ".join(seq) for seq in sequence_emb]
        bert_tokeniser = BertTokenizer.from_pretrained(
            "Exscientia/IgBert_unpaired", do_lower_case=False
        )
        bert_model = BertModel.from_pretrained(
            "Exscientia/IgBert_unpaired", add_pooling_layer=False
        ).to(device)  # Move to GPU
        tokens = bert_tokeniser.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding="max_length",
            max_length=286,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}  # Move tokens to GPU
        with torch.no_grad():
            output = bert_model(
                input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
            )
            bert_residue_embeddings = output.last_hidden_state
            bert_residue_embeddings = bert_residue_embeddings[:, 1:, :]
        return bert_residue_embeddings.cpu()
    device = get_device(gpu)
    paired_sequences = []
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb, strict=False):
        paired_sequences.append(" ".join(seq_heavy) + " [SEP] " + " ".join(seq_light))
    bert_tokeniser = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
    bert_model = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False).to(
        device
    )  # Move to GPU
    tokens = bert_tokeniser.batch_encode_plus(
        paired_sequences,
        add_special_tokens=True,
        padding="max_length",
        max_length=287,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}  # Move tokens to GPU
    with torch.no_grad():
        output = bert_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        bert_residue_embeddings = output.last_hidden_state
    lenghts_heavy = [len(seq_heavy) for seq_heavy in sequence_heavy_emb]
    bert_residue_embeddings = torch.stack(
        [
            torch.cat([each[1 : len_heavy + 1, :], each[len_heavy + 2 :, :]], dim=0)
            for (each, len_heavy) in zip(bert_residue_embeddings, lenghts_heavy, strict=False)
        ]
    )
    return bert_residue_embeddings.cpu()  # Move back to CPU


def compute_igt5_embeddings(
    sequence_heavy_emb: list,
    sequence_light_emb: list,
    gpu: int = 0,
    *,
    single_chain: bool = False,
) -> torch.Tensor:
    """Compute igt5 embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.
        gpu (int): Gpu to use.
        single_chain (bool): Single chain mode.

    Returns
    -------
        torch.Tensor: IgT5 embeddings.
    """
    if single_chain:
        sequence_emb = sequence_heavy_emb if len(sequence_heavy_emb) > 0 else sequence_light_emb
        device = get_device(gpu)
        sequences = [" ".join(seq) for seq in sequence_emb]
        igt5_tokeniser = T5Tokenizer.from_pretrained(
            "Exscientia/IgT5_unpaired", do_lower_case=False
        )
        igt5_model = T5EncoderModel.from_pretrained("Exscientia/IgT5_unpaired").to(
            device
        )  # Move to GPU
        tokens = igt5_tokeniser.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding="max_length",
            max_length=286,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}  # Move tokens to GPU
        with torch.no_grad():
            output = igt5_model(
                input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
            )
            igt5_residue_embeddings = output.last_hidden_state
            igt5_residue_embeddings = igt5_residue_embeddings[:, 1:, :]
        return igt5_residue_embeddings.cpu()  # Move back to CPU
    device = get_device(gpu)
    paired_sequences = []
    for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb, strict=False):
        paired_sequences.append(" ".join(seq_heavy) + " </s> " + " ".join(seq_light))
    igt5_tokeniser = T5Tokenizer.from_pretrained("Exscientia/IgT5", do_lower_case=False)
    igt5_model = T5EncoderModel.from_pretrained("Exscientia/IgT5").to(device)  # Move to GPU
    tokens = igt5_tokeniser.batch_encode_plus(
        paired_sequences,
        add_special_tokens=True,
        padding="max_length",
        max_length=287,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}  # Move tokens to GPU
    with torch.no_grad():
        output = igt5_model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        igt5_residue_embeddings = output.last_hidden_state
    lenghts_heavy = [len(seq_heavy) for seq_heavy in sequence_heavy_emb]
    igt5_residue_embeddings = torch.stack(
        [
            torch.cat([each[1 : len_heavy + 1, :], each[len_heavy + 2 :, :]], dim=0)
            for (each, len_heavy) in zip(igt5_residue_embeddings, lenghts_heavy, strict=False)
        ]
    )
    return igt5_residue_embeddings.cpu()  # Move back to CPU


def compute_t5_embeddings(
    sequence_heavy_emb: list,
    sequence_light_emb: list,
    gpu: int = 0,
    *,
    single_chain: bool = False,  # noqa: ARG001
) -> torch.Tensor:
    """Compute prot-t5 embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.
        gpu (int): Gpu to use.
        single_chain (bool): Single chain mode.

    Returns
    -------
        torch.Tensor: Prot-T5 embeddings.
    """
    device = get_device(gpu)
    prot_t5_tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    prot_t5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(
        device
    )  # Move to GPU

    prot_t5_sequences = [
        "".join(seq_heavy) + "".join(seq_light)
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb, strict=False)
    ]
    prot_t5_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in prot_t5_sequences]
    prot_t5_ids = prot_t5_tokenizer(
        prot_t5_sequences,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    prot_t5_ids = {
        key: value.to(device) for key, value in prot_t5_ids.items()
    }  # Move tokens to GPU

    with torch.no_grad():
        prot_t5_output = prot_t5_model(
            input_ids=prot_t5_ids["input_ids"], attention_mask=prot_t5_ids["attention_mask"]
        )

    prot_t5_embeddings = prot_t5_output.last_hidden_state
    pad_length = 285 - prot_t5_embeddings.size(1)
    padding = (0, 0, 0, pad_length)
    prot_t5_embeddings = F.pad(prot_t5_embeddings, padding, mode="constant", value=0)
    return prot_t5_embeddings.cpu()  # Move back to CPU


def compute_prost_t5_embeddings(
    sequence_heavy_emb: list,
    sequence_light_emb: list,
    gpu: int = 0,
    *,
    single_chain: bool = False,  # noqa: ARG001
) -> torch.Tensor:
    """Compute Prost-T5 embeddings.

    Args:
        sequence_heavy_emb (List): Heavy sequences.
        sequence_light_emb (List): Light sequences.
        gpu (int): GPU index to use.
        single_chain (bool): Single chain mode.

    Returns
    -------
        torch.Tensor: Prost-T5 embeddings.
    """
    device = get_device(gpu)
    # Load Prost-T5
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
    # Build sequences
    prost_sequences = [
        "".join(seq_heavy) + "".join(seq_light)
        for seq_heavy, seq_light in zip(sequence_heavy_emb, sequence_light_emb, strict=False)
    ]
    prost_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in prost_sequences]
    prost_sequences = [
        "<AA2fold>" + " " + s
        if s.isupper()
        else "<fold2AA>" + " " + s  # this expects 3Di sequences to be already lower-case
        for s in prost_sequences
    ]
    # Tokenize
    ids = tokenizer.batch_encode_plus(
        prost_sequences, add_special_tokens=True, padding="longest", return_tensors="pt"
    ).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(ids.input_ids, attention_mask=ids.attention_mask)

    embeddings = embedding_repr.last_hidden_state

    # REMOVE FIRST TOKEN → Prost-T5 canonical postprocessing
    embeddings = embeddings[:, 1:, :]

    # Pad to 285 tokens
    pad_length = 285 - embeddings.size(1)
    padding = (0, 0, 0, pad_length)
    embeddings = F.pad(embeddings, padding, mode="constant", value=0)
    return embeddings.cpu()


def save_embedding_pt(name, emb, folder):
    """Save full and PCA-128 reduced embeddings as .pt tensors."""
    full_path = folder / Path(f"{name}.pt")
    torch.save(emb, full_path)


@app.command()
def main(
    dict_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path of csv file to use for pdb list.",
        show_default=False,
    ),
    emb_proc_size: int = typer.Option(
        100,
        "--emb-proc-size",
        help=(
            "We create embeddings chunk by chunk to avoid memory explosion. "
            "This is the chunk size. Optimal value depends on your computer."
        ),
    ),
    gpu: int = typer.Option(
        0,
        "--gpu",
        help="Choose index of GPU device to use if multiple GPUs available. By default it's the"
        "first one (index 0). -1 forces cpu usage. If no GPU is available, CPU is used.",
    ),
    single_chain: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--single-chain",
        help=(
            "Generate emebddings using llms on single chain mode, which slightly increases"
            "performance. Default to False."
        ),
    ),
) -> None:
    """Create dataset to train the neural network.

    Sequences and labels are saved in a .json file, and LPLM embeddings are saved in a .pt file.
    """
    log.info("LOADING DICTIONARY")
    save_folder = dict_path.parents[0]
    with dict_path.open(encoding="utf-8") as f:
        dataset_dict = json.load(f)

    sequence_heavy_emb = [dataset_dict[index]["H_id sequence"] for index in dataset_dict]
    sequence_light_emb = [dataset_dict[index]["L_id sequence"] for index in dataset_dict]

    log.info("CREATING EMBEDDINGS", embedding_model="ESM")
    esm_embeddings = process_batch(
        compute_esm_embeddings,
        sequence_heavy_emb,
        sequence_light_emb,
        emb_proc_size,
        gpu=gpu,
    )
    save_embedding_pt("esm_embeddings", esm_embeddings, save_folder)

    log.info("CREATING EMBEDDINGS", embedding_model="IgT5")
    igt5_embeddings = process_batch(
        compute_igt5_embeddings,
        sequence_heavy_emb,
        sequence_light_emb,
        emb_proc_size,
        single_chain=single_chain,
        gpu=gpu,
    )
    print(igt5_embeddings.shape)
    save_embedding_pt("igT5_embeddings", igt5_embeddings, save_folder)

    log.info("CREATING EMBEDDINGS", embedding_model="IgBert")
    bert_embeddings = process_batch(
        compute_igbert_embeddings,
        sequence_heavy_emb,
        sequence_light_emb,
        emb_proc_size,
        single_chain=single_chain,
        gpu=gpu,
    )
    save_embedding_pt("igbert_embeddings", bert_embeddings, save_folder)

    log.info("CREATING EMBEDDINGS", embedding_model="Ablang2")
    ablang_embeddings = process_batch(
        compute_ablang_embeddings,
        sequence_heavy_emb,
        sequence_light_emb,
        emb_proc_size,
        gpu=gpu,
    )
    save_embedding_pt("ablang2_embeddings", ablang_embeddings, save_folder)

    log.info("CREATING EMBEDDINGS", embedding_model="Antiberty")
    antiberty_embeddings = process_batch(
        compute_antiberty_embeddings,
        sequence_heavy_emb,
        sequence_light_emb,
        emb_proc_size,
        gpu=gpu,
    )
    save_embedding_pt("antiberty_embeddings", antiberty_embeddings, save_folder)

    log.info("CREATING EMBEDDINGS", embedding_model="Prot-T5")
    prot_t5_embeddings = process_batch(
        compute_t5_embeddings,
        sequence_heavy_emb,
        sequence_light_emb,
        emb_proc_size,
        gpu=gpu,
    )
    save_embedding_pt("prot-t5_embeddings", prot_t5_embeddings, save_folder)


if __name__ == "__main__":
    app()
