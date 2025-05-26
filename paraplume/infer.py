"""Create dictionary of sequence/labels and corresponding LLM embeddings."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from torch.nn import Dropout, Linear, ReLU, Sequential, Sigmoid

from paraplume.create_dataset import get_llm_to_embedding_dict
from paraplume.utils import get_logger

app = typer.Typer(add_completion=False)
warnings.filterwarnings("ignore")

log = get_logger()

# pylint: disable=E1101


def process_embedding(llm: str, emb: torch.Tensor, heavy: int, light: int) -> torch.Tensor:
    """Return tensor of embedding given lengths of heavy and light chains and embeddings to use.

    Args:
        embedding (torch.Tensor): Total embedding in which to do selection.
        embedding_models (List[str]): List of embeddings to use.
        heavy (int): Heavy chain length.
        light (int): Light chain length.

    Returns
    -------
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
    return emb[ran_aa, :]


@app.command()
def predict_from_df( # noqa: PLR0913,PLR0915
    file_path: Path = typer.Argument( # noqa: B008
        ...,
        help="Path of the file.",
        show_default=False,
    ),
    result_folder: Path = typer.Argument( # noqa: B008
        ...,
        help="Path of the folder where the model is saved.",
        show_default=False,
    ),
    name: str = typer.Option(
        "",
        "--name",
        help="Extension to add to the file.",
    ),
    gpu: int = typer.Option(1, "--gpu", help="Which GPU to use."),
    compute_embeddings: bool = typer.Option( # noqa: FBT001
        False, # noqa: FBT003
        "--compute-embeddings",
        help="Compute paratope and classical embeddings for each sequence.",
    ),
    emb_proc_size: int = typer.Option(
        100,
        "--emb-proc-size",
        help="We create embeddings batch by batch to avoid memory explosion. This is the batch\
            size. Optimal value depends on your computer. Defaults to 100.",
    ),
    single_chain: bool = typer.Option( # noqa: FBT001
        False, # noqa: FBT003
        "--single-chain",
        help="Chain to use for single chain model. Defaults to heavy.",
    ),
) -> None:
    """Predict paratope from sequence."""
    summary_dict_path = result_folder / Path("summary_dict.json")
    log.info("Loading training summary dictionary.", path=summary_dict_path.as_posix())
    with summary_dict_path.open(encoding="utf-8") as f:
        summary_dict = json.load(f)

    layers = []
    input_size = int(summary_dict["input_size"])
    dims = [int(each) for each in summary_dict["dims"].split(",")]
    dropouts = [0] * len(dims)
    for i, _ in enumerate(dims):
        if i == 0:
            layers.append(Linear(input_size, dims[i]))
            layers.append(Dropout(dropouts[i]))
            layers.append(ReLU())
        else:
            layers.append(Linear(dims[i - 1], dims[i]))
            layers.append(Dropout(dropouts[i]))
            layers.append(ReLU())
    model = Sequential(Sequential(*layers), Sequential(Linear(dims[-1], 1), Sigmoid()))
    model_path = result_folder / Path("checkpoint.pt")
    log.info("Loading model.", path=model_path.as_posix())
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    llm_models = summary_dict["embedding_models"]
    if llm_models == "all":
        llm_models = "ablang2,igbert,igT5,esm,antiberty,prot-t5"
    llm_list = llm_models.split(",")

    df = pd.read_csv(file_path)
    df["sequence_heavy"] = df["sequence_heavy"].fillna("")
    df["sequence_light"] = df["sequence_light"].fillna("")
    sequences_heavy = df["sequence_heavy"].tolist()
    sequences_light = df["sequence_light"].tolist()
    llm_to_embedding_dict = get_llm_to_embedding_dict(
        sequences_heavy,
        sequences_light,
        emb_proc_size=emb_proc_size,
        llm_list=llm_list,
        single_chain=single_chain,
        gpu=gpu,
    )

    heavy_outputs = []
    light_outputs = []
    embeddings_classical = []
    embeddings_paratope = []
    for i, (seq_heavy, seq_light) in enumerate(zip(sequences_heavy, sequences_light, strict=False)):
        heavy, light = len(seq_heavy), len(seq_light)
        emb_llm_list_i = []
        for llm, emb_llm in llm_to_embedding_dict.items():
            emb_processed_i_llm = process_embedding(
                llm=llm,
                emb=emb_llm[i],
                heavy=heavy,
                light=light,
            )
            emb_llm_list_i.append(emb_processed_i_llm)
        emb_processed_i = torch.cat(emb_llm_list_i, dim=-1)
        emb = emb_processed_i.to(device)
        output = np.round(
            model(emb).cpu().detach().numpy().flatten().astype(np.float64), 12
        ).tolist()
        heavy_outputs.append(output[:heavy])
        light_outputs.append(output[heavy:])

        if not compute_embeddings:
            continue
        emb_classical = emb.sum(0) / emb.shape[0]
        emb_classical = np.round(
            emb_classical.cpu().detach().numpy().flatten().astype(np.float64), 12
        ).tolist()
        embeddings_classical.append(emb_classical)
        emb_paratope = np.zeros(int(emb.shape[-1]))
        normalized_output = output / np.sum(output)
        for prob, embed in zip(normalized_output, emb, strict=False):
            emb_paratope += prob * embed.cpu().detach().numpy()
        emb_paratope = np.round(emb_paratope.flatten().astype(np.float64), 12).tolist()
        embeddings_paratope.append(emb_paratope)

    if compute_embeddings:
        df["embeddings_paratope"] = embeddings_paratope
        df["embeddings_classical"] = embeddings_classical
    df["model_prediction_heavy"] = heavy_outputs
    df["model_prediction_light"] = light_outputs
    result_path = file_path.parents[0] / Path(f"{name}paratope_" + file_path.name)
    df.to_csv(result_path)
    return output


@app.command()
def predict_from_sequence(
    sequence_heavy: str = typer.Argument(
        ...,
        help="Path of csv file to use for pdb list.",
        show_default=False,
    ),
    sequence_light: str = typer.Argument(
        ...,
        help="Path of csv file to use for pdb list.",
        show_default=False,
    ),
    result_folder: Path = typer.Argument( # noqa: B008
        ...,
        help="Path of the result folder.",
        show_default=False,
    ),
    single_chain: bool = typer.Option( # noqa: FBT001
        False, # noqa: FBT003
        "--heavy-chain",
        help="Chain to use for single chain model. Defaults to heavy.",
    ),
    gpu: int = typer.Option(1, "--gpu", help="Which GPU to use."),
) -> None:
    """Predict paratope from sequence."""
    summary_dict_path = result_folder / Path("summary_dict.json")
    log.info("Loading training summary dictionary.", path=summary_dict_path.as_posix())
    with summary_dict_path.open(encoding="utf-8") as f:
        summary_dict = json.load(f)

    layers = []
    input_size = int(summary_dict["input_size"])
    dims = [int(each) for each in summary_dict["dims"].split(",")]
    dropouts = [0] * len(dims)
    for i, _ in enumerate(dims):
        if i == 0:
            layers.append(Linear(input_size, dims[i]))
            layers.append(Dropout(dropouts[i]))
            layers.append(ReLU())
        else:
            layers.append(Linear(dims[i - 1], dims[i]))
            layers.append(Dropout(dropouts[i]))
            layers.append(ReLU())
    model = Sequential(Sequential(*layers), Sequential(Linear(dims[-1], 1), Sigmoid()))
    model_path = result_folder / Path("checkpoint.pt")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    llm_models = summary_dict["embedding_models"]
    if llm_models == "all":
        llm_models = "ablang2,igbert,igT5,esm,antiberty,prot-t5"
    llm_list = llm_models.split(",")
    llm_to_embedding_dict = get_llm_to_embedding_dict(
        [sequence_heavy],
        [sequence_light],
        emb_proc_size=1,
        llm_list=llm_list,
        single_chain=single_chain,
        gpu=gpu,
    )
    heavy, light = len(sequence_heavy), len(sequence_light)
    embeddings_processed_list = []
    for llm, emb in llm_to_embedding_dict.items():
        emb_sequence = emb[0]
        emb_processed = process_embedding(
            llm=llm,
            emb=emb_sequence,
            heavy=heavy,
            light=light,
        )
        embeddings_processed_list.append(emb_processed)
    embeddings_processed = torch.cat(embeddings_processed_list, dim=-1)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    embeddings_processed = embeddings_processed.to(device)
    output = model(embeddings_processed).cpu().detach().numpy().flatten()
    print("===== Heavy Chain =====")
    for each in zip(sequence_heavy, output[:heavy], strict=False):
        print(f"{each[0]}  --> {np.round(float(each[1]),3)}")
    print("===== Light Chain =====")
    for each in zip(sequence_light, output[heavy : heavy + light], strict=False):
        print(f"{each[0]}  --> {np.round(float(each[1]),3)}")
    return output


if __name__ == "__main__":
    app()
