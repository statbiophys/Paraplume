"""Create dictionary of sequence/labels and corresponding LLM embeddings."""
import json
import warnings
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from torch.nn import Dropout, Linear, ReLU, Sequential, Sigmoid

from paraplume.create_dataset import get_llm_to_embedding_dict
from paraplume.utils import get_device, get_logger

app = typer.Typer(add_completion=False)
warnings.filterwarnings("ignore")

log = get_logger()

LLM_DIM_DICT = {
    "ablang2": list(range(2048, 2528)),
    "igT5": list(range(1024, 2048)),
    "igbert": list(range(1024)),
    "esm": list(range(2528, 3808)),
    "antiberty": list(range(3808, 4320)),
    "prot-t5": list(range(4320, 5344)),
    "all": list(range(5344)),
}

def process_embedding(llm: str, emb: torch.Tensor, heavy: int, light: int) -> torch.Tensor:
    """Return unpadded tensor of the llm's embedding given lengths of heavy and light chains.

    Args:
        llm (str): LLM to use.
        emb (torch.Tensor): Padded embedding for this llm.
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

def predict_paratope_seq(  # noqa: PLR0913
    sequence_heavy: str ="",
    sequence_light: str ="",
    custom_model: Path | None = None,
    gpu: int = 0,
    *,
    large: bool = True,
    single_chain: bool = False,
) -> tuple:
    """Predict paratope given two sequence chains.

    Args:
        sequence_heavy (str | None, optional): Heavy chain sequence. Defaults to None.
        sequence_light (str | None, optional): Light chain sequence. Defaults to None.
        custom_model (Path | None, optional): Use custom model folder. Defaults to None.
        gpu (int, optional): Gpu to use. Defaults to 0.
        large (bool, optional): Use model trained on 6 embeddings. Defaults to True.
        single_chain (bool, optional): Compute embeddings using LLMs trained on single chains.\
            Defaults to False.

    Returns
    -------
        tuple: _description_
    """
    if not custom_model:
        subfolder = "large" if large else "small"
        with resources.as_file(
            resources.files("paraplume.trained_models") / subfolder
        ) as model_path:
            custom_model = model_path
    summary_dict_path = custom_model / Path("summary_dict.json")
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
    model_path = custom_model / Path("checkpoint.pt")
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
    device = get_device(gpu)
    model = model.to(device)
    embeddings_processed = embeddings_processed.to(device)
    output = model(embeddings_processed).cpu().detach().numpy().flatten().tolist()
    return output[:heavy], output[heavy : heavy + light]

def predict_paratope( # noqa: PLR0913,PLR0915
    df:pd.DataFrame,
    custom_model: Path | None = None,
    gpu: int = 0,
    emb_proc_size: int = 100,
    *,
    compute_sequence_embeddings: bool = False,
    single_chain: bool = False,
    large: bool = True,
)->pd.DataFrame:
    """Predict the paratope for sequences in dataframe df.

    Args:
        df (pd.DataFrame): Input dataframe.
        custom_model (Path | None, optional): Use custom model folder. Defaults to None.
        gpu (int, optional): Gpu to use. Defaults to 0.
        emb_proc_size (int, optional): Compute embeddings by batch of size 'emb_proc_size'.\
            Defaults to 100.
        compute_sequence_embeddings (bool, optional): Compute paratope and classical sequence\
            embeddings for each sequence and llm. Only possible when using the default\
                trained_models/large.
            Defaults to False.
        single_chain (bool, optional): Compute embeddings using LLMs trained on single chains.\
            Defaults to False.
        large (bool, optional): Use model trained on 6 embeddings. Defaults to True.

    Raises
    ------
        ValueError: If trying to compute sequence embeddings when using default or small model.

    Returns
    -------
        pd.DataFrame: Dataframe with paratope predictions as new columns.
    """
    if (custom_model or not large) and compute_sequence_embeddings:
        msg = "Sequence embedding computation only possible when using default large trained model."
        raise ValueError(msg)
    if not custom_model:
        subfolder = "large" if large else "small"
        with resources.as_file(
            resources.files("paraplume.trained_models") / subfolder
        ) as model_path:
            custom_model = model_path
    summary_dict_path = custom_model / Path("summary_dict.json")
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
    model_path = custom_model / Path("checkpoint.pt")
    log.info("Loading model.", path=model_path.as_posix())
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    device = get_device(gpu)
    model = model.to(device)

    llm_models = summary_dict["embedding_models"]
    if llm_models == "all":
        llm_models = "ablang2,igbert,igT5,esm,antiberty,prot-t5"
    llm_list = llm_models.split(",")

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

        if not compute_sequence_embeddings:
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

    if compute_sequence_embeddings:
        df["embeddings_paratope"] = embeddings_paratope
        df["embeddings_classical"] = embeddings_classical
        for llm in llm_list:
            llm_range = LLM_DIM_DICT[llm]
            embeddings_llm_paratope = np.vstack(embeddings_paratope)[:, llm_range].tolist()
            embeddings_llm_classical = np.vstack(embeddings_classical)[:, llm_range].tolist()
            df[f"{llm}_paratope_seq_emb"] = embeddings_llm_paratope
            df[f"{llm}_classical_seq_emb"] = embeddings_llm_classical

    df["model_prediction_heavy"] = heavy_outputs
    df["model_prediction_light"] = light_outputs
    return df

@app.command()
def file_to_paratope(  # noqa: PLR0913
    file_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path of the csv file.",
        show_default=False,
    ),
    custom_model: Path
    | None = typer.Option(  # noqa: B008
        None,
        "--custom-model",
        help="Custom trained model folder path to do inference. Needs to contain the same files \
            as paraplume/trained_models/large which is the output of a paraplume.train ",
    ),
    name: str = typer.Option(
        "paratope_",
        "--name",
        help="Prefix to add to the file.",
    ),
    gpu: int = typer.Option(0, "--gpu", help="Which GPU to use."),
    emb_proc_size: int = typer.Option(
        100,
        "--emb-proc-size",
        help="We create embeddings batch by batch to avoid memory explosion. This is the batch\
            size. Optimal value depends on your computer. Defaults to 100.",
    ),
    compute_sequence_embeddings: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--compute-sequence-embeddings",
        help="Compute paratope and classical sequence embeddings for each sequence and llm.\
            Only possible when using the default trained_models/large.",
    ),
    single_chain: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--single-chain",
        help="Infer paratope on single chain data. Default to False.",
    ),
    large: bool = typer.Option(  # noqa: FBT001
        True,  # noqa: FBT003
        "--large/--small",
        help="Use default Paraplume or the smallest version using only ESM-2 embeddings.",
    ),
) -> pd.DataFrame:
    """Predict paratope from sequence."""
    df = pd.read_csv(file_path)
    predict_paratope(
        df,
        custom_model=custom_model,
        gpu=gpu,
        emb_proc_size=emb_proc_size,
        compute_sequence_embeddings=compute_sequence_embeddings,
        single_chain=single_chain,
        large=large,
    )
    result_path = file_path.parents[0] / Path(f"{name}" + file_path.stem)
    df.to_pickle(result_path.with_suffix(".pkl"))
    return df


@app.command()
def seq_to_paratope(
    sequence_heavy: str  = typer.Option(
        "",
        "--heavy-chain",
        "-h",
        help="Heavy chain amino acid sequence.",
        show_default=False,
    ),
    sequence_light: str= typer.Option(
        "",
        "--light-chain",
        "-l",
        help="Light chain amino acid sequence.",
        show_default=False,
    ),
    custom_model: Path
    | None = typer.Option(  # noqa: B008
        None,
        "--custom-model",
        help=(
            "Custom trained model folder path to do inference. Needs to contain the same files "
            "as paraplume/trained_models/large which is the output of a training phase."
            )
    ),
    gpu: int = typer.Option(0, "--gpu", help="Which GPU to use."),
    large: bool = typer.Option(  # noqa: FBT001
        True,  # noqa: FBT003
        "--large/--small",
        help="Use default Paraplume or the smallest version using only ESM-2 embeddings.",
    ),
) -> None:
    """Predict paratope from sequence."""
    single_chain=(sequence_heavy is None) or (sequence_light is None)
    output_heavy, output_light = predict_paratope_seq(sequence_heavy=sequence_heavy,
                                                    sequence_light=sequence_light,
                                                    custom_model=custom_model,
                                                    gpu=gpu,
                                                    large=large,
                                                    single_chain=single_chain)
    if output_heavy:
        print("===== Heavy Chain =====")
        print(f"{'AA':<4}  {'Probability':>10}")
        print("-" * 20)
        for aa, prob in zip(sequence_heavy, output_heavy, strict=False):
            print(f"{aa:<4}  --> {np.round(float(prob), 3):>8.3f}")
    if output_light:
        print("\n===== Light Chain =====")
        print(f"{'AA':<4}  {'Probability':>10}")
        print("-" * 20)
        for aa, prob in zip(sequence_light, output_light, strict=False):
            print(f"{aa:<4}  --> {np.round(float(prob), 3):>8.3f}")

if __name__ == "__main__":
    app()
