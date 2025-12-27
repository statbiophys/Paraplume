"""Create dictionary of sequence/labels and corresponding LLM embeddings."""

import json
import warnings
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer

from paraplume.create_embeddings import (
    compute_ablang_embeddings,
    compute_antiberty_embeddings,
    compute_esm_embeddings,
    compute_igbert_embeddings,
    compute_igt5_embeddings,
    compute_t5_embeddings,
    process_batch,
)
from paraplume.shapp_utils import (
    PLM_EMBEDDING_DIMENSIONS,
    compute_shap_importance,
    plot_residue_importance,
)
from paraplume.utils import build_model, get_device, get_logger

app = typer.Typer(add_completion=False)
warnings.filterwarnings("ignore")

log = get_logger()

llm_to_func = {
    "ablang2": compute_ablang_embeddings,
    "igT5": compute_igt5_embeddings,
    "igbert": compute_igbert_embeddings,
    "esm": compute_esm_embeddings,
    "antiberty": compute_antiberty_embeddings,
    "prot-t5": compute_t5_embeddings,
}


def analyze_shap_for_dataframe(  # noqa: PLR0913,PLR0915
    df: pd.DataFrame,
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    llm_list: list[str],
    background_heavy: torch.Tensor,
    background_light: torch.Tensor,
    output_dir: Path,
):
    """
    Perform SHAP analysis on a dataframe of sequences.

    Args:
        df: DataFrame with sequences and predictions
        model: Trained model
        embeddings: All embeddings for sequences
        llm_list: List of PLM names
        background_heavy: Background embeddings for heavy chain SHAP
        background_light: Background embeddings for light chain SHAP
        output_dir: Directory to save results
    """
    model = model.cpu()
    embeddings = embeddings.cpu()
    log.info("Creating SHAP results output folder", output_dir=output_dir.as_posix())
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get PLM dimensions
    plm_dimensions = [PLM_EMBEDDING_DIMENSIONS[plm] for plm in llm_list]

    # Get sequence identifiers
    sequence_ids = [f"seq_{i}" for i in range(len(df))]
    if "sequence_id" in df.columns:
        sequence_ids = df["sequence_id"].tolist()

    # Compute lengths
    sequences_heavy = df["sequence_heavy"].fillna("").tolist()
    sequences_light = df["sequence_light"].fillna("").tolist()
    heavy_lens = [len(seq) for seq in sequences_heavy]
    light_lens = [len(seq) for seq in sequences_light]

    # Separate importance scores for heavy and light chains
    heavy_importance_per_sequence = []
    light_importance_per_sequence = []

    log.info("Computing SHAP importance for each sequence...")
    for i in range(len(df)):
        # Process heavy chain
        if heavy_lens[i] == 0:
            heavy_importance_per_sequence.append([0.0] * len(llm_list))
            continue
        heavy_embedding = embeddings[i, : heavy_lens[i], :]
        heavy_importance = compute_shap_importance(
            model, heavy_embedding, plm_dimensions, background_heavy
        )
        heavy_importance_per_sequence.append(heavy_importance)
    for i in range(len(df)):
        if light_lens[i] == 0:
            light_importance_per_sequence.append([0.0] * len(llm_list))
            continue
        light_start = heavy_lens[i]
        light_end = heavy_lens[i] + light_lens[i]
        light_embedding = embeddings[i, light_start:light_end, :]
        light_importance = compute_shap_importance(
            model, light_embedding, plm_dimensions, background_light
        )
        light_importance_per_sequence.append(light_importance)

    # 1. Create individual heatmaps and residue plots for each sequence (separate heavy/light)
    log.info("Generating per-sequence visualizations...")
    for i in range(len(df)):
        if heavy_lens[i] == 0:
            continue
        seq_id = sequence_ids[i]
        seq_heavy = sequences_heavy[i]
        pred_heavy = df["model_prediction_heavy"].iloc[i]

        heavy_embedding = embeddings[i, : heavy_lens[i], :]
        # Compute per-residue importance for heavy chain
        heavy_residue_importance = []
        for j in range(heavy_lens[i]):
            residue_embedding = heavy_embedding[j : j + 1, :]
            importance = compute_shap_importance(
                model, residue_embedding, plm_dimensions, background_heavy
            )
            heavy_residue_importance.append(importance)
        heavy_residue_matrix = np.array(heavy_residue_importance)
        plot_residue_importance(
            labels=list(pred_heavy),
            amino_acids=list(seq_heavy),
            importance_matrix=heavy_residue_matrix,
            plm_names=llm_list,
            save_path=output_dir / f"{seq_id}_heavy_chain_residue_importance.png",
        )
    for i in range(len(df)):
        if light_lens[i] == 0:
            continue
        seq_id = sequence_ids[i]
        seq_light = sequences_light[i]
        pred_light = df["model_prediction_light"].iloc[i]

        light_start = heavy_lens[i]
        light_end = heavy_lens[i] + light_lens[i]
        light_embedding = embeddings[i, light_start:light_end, :]

        # Compute per-residue importance for light chain
        light_residue_importance = []
        for j in range(light_lens[i]):
            residue_embedding = light_embedding[j : j + 1, :]
            importance = compute_shap_importance(
                model, residue_embedding, plm_dimensions, background_light
            )
            light_residue_importance.append(importance)
        light_residue_matrix = np.array(light_residue_importance)
        plot_residue_importance(
            labels=list(pred_light),
            amino_acids=list(seq_light),
            importance_matrix=light_residue_matrix,
            plm_names=llm_list,
            save_path=output_dir / f"{seq_id}_light_chain_residue_importance.png",
            rotation=90,
        )
    log.info("SHAP analysis complete!")


def predict_paratope(  # noqa: PLR0913,PLR0915
    df: pd.DataFrame,
    custom_model: Path | None = None,
    gpu: int = 0,
    emb_proc_size: int = 100,
    *,
    compute_sequence_embeddings: bool = False,
    single_chain: bool = False,
    large: bool = True,
    compute_shap: bool = False,
    shap_output_dir: Path = Path("./shap_results/"),
) -> pd.DataFrame:
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
        compute_shap (bool, optional): Compute SHAP importance analysis. Defaults to False.
        shap_output_dir (Path): Directory to save SHAP results.

    Raises
    ------
        ValueError: If trying to compute sequence embeddings when using default or small model.

    Returns
    -------
        pd.DataFrame: Dataframe with paratope predictions as new columns.
    """
    device = get_device(gpu)
    if not custom_model:
        subfolder = "large_1_0" if large else "small_1_0"
        with resources.as_file(
            resources.files("paraplume.trained_models") / subfolder
        ) as model_path:
            custom_model = model_path
    summary_dict_path = custom_model / Path("summary_dict.json")
    log.info("Loading training summary dictionary.", path=summary_dict_path.as_posix())
    with summary_dict_path.open(encoding="utf-8") as f:
        summary_dict = json.load(f)

    input_size = int(summary_dict["input_size"])
    dims = [int(each) for each in summary_dict["dims"].split(",")]
    dropouts = [0] * len(dims)

    model_path = custom_model / Path("checkpoint.pt")
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    nested = any("0.0." in key for key in state_dict.keys())

    model = build_model(input_size, dims, dropouts, nested=nested)
    log.info("Loading model.", path=model_path.as_posix())
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    llm_models = summary_dict["embedding_models"]
    if llm_models == "all":
        llm_models = "ablang2,igbert,igT5,esm,antiberty,prot-t5"
    llm_list = llm_models.split(",")

    df["sequence_heavy"] = df["sequence_heavy"].fillna("")
    df["sequence_light"] = df["sequence_light"].fillna("")
    sequences_heavy = df["sequence_heavy"].tolist()
    sequences_light = df["sequence_light"].tolist()

    seq_emb_list = []
    for llm in llm_list:
        seq_emb_list.append(
            process_batch(
                llm_to_func[llm],
                sequences_heavy,
                sequences_light,
                emb_proc_size,
                gpu=gpu,
                single_chain=single_chain,
            )
        )
    embeddings = torch.cat(seq_emb_list, dim=-1).to(device)
    device = embeddings.device  # keep the same device

    # Compute lengths of each sequence
    heavy_lens = [len(seq) for seq in sequences_heavy]
    light_lens = [len(seq) for seq in sequences_light]
    total_lens = [heavy + light for heavy, light in zip(heavy_lens, light_lens, strict=False)]

    # Remove padding → list of (Ni, 2304)
    embedding_list = [embeddings[i, : total_lens[i], :] for i in range(len(total_lens))]
    heavy_outputs = []
    light_outputs = []
    for i, emb in enumerate(embedding_list):
        # Forward pass → shape: (Ni,)
        preds = model(emb).cpu().detach().numpy().flatten()
        heavy_outputs.append(preds[: heavy_lens[i]])
        light_outputs.append(preds[heavy_lens[i] : heavy_lens[i] + light_lens[i]])

    df["model_prediction_heavy"] = heavy_outputs
    df["model_prediction_light"] = light_outputs

    # SHAP Analysis
    if compute_shap:
        background_path = resources.files("paraplume.background")
        background_heavy = torch.cat(
            [
                torch.load(background_path / Path(f"{model}_heavy.pt"), weights_only=True)
                for model in llm_list
            ],
            dim=-1,
        )

        background_light = torch.cat(
            [
                torch.load(background_path / Path(f"{model}_light.pt"), weights_only=True)
                for model in llm_list
            ],
            dim=-1,
        )

        log.info("Starting SHAP analysis...")
        analyze_shap_for_dataframe(
            df=df,
            model=model,
            embeddings=embeddings,
            llm_list=llm_list,
            background_heavy=background_heavy,
            background_light=background_light,
            output_dir=shap_output_dir,
        )

    if not compute_sequence_embeddings:
        return df

    llm_shapes = np.cumsum([0] + [llm.shape[-1] for llm in seq_emb_list])
    embeddings_classical = []
    embeddings_paratope = []

    # Join heavy + light predictions back together (same order as embedding_list)
    outputs_joint = [
        torch.tensor(np.concatenate([heavy, light], axis=0), device=device)
        for heavy, light in zip(heavy_outputs, light_outputs, strict=False)
    ]

    for _, (emb, probs) in enumerate(zip(embedding_list, outputs_joint, strict=False)):
        # Classical → mean over sequence length
        emb_classical = emb.mean(dim=0)
        embeddings_classical.append(
            emb_classical.cpu().detach().numpy().astype(np.float64).round(12).tolist()
        )

        # Paratope-weighted → normalize probabilities then weighted sum
        probs_normalized = probs / probs.sum()
        emb_paratope = torch.sum(emb * probs_normalized[:, None], dim=0)
        embeddings_paratope.append(
            emb_paratope.cpu().detach().numpy().astype(np.float64).round(12).tolist()
        )

    df["embeddings_paratope"] = embeddings_paratope
    df["embeddings_classical"] = embeddings_classical

    np_embeddings_paratope = np.array(embeddings_paratope)
    np_embeddings_classical = np.array(embeddings_classical)

    for i in range(len(llm_shapes) - 1):
        llm_range = list(range(llm_shapes[i], llm_shapes[i + 1]))
        llm = llm_list[i]
        llm_embedding_paratope = np_embeddings_paratope[:, llm_range].tolist()
        llm_embedding_classical = np_embeddings_classical[:, llm_range].tolist()
        df[f"{llm}_paratope_seq_emb"] = llm_embedding_paratope
        df[f"{llm}_classical_seq_emb"] = llm_embedding_classical

    return df


@app.command()
def file_to_paratope(  # noqa: PLR0913
    file_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path of the csv file.",
        show_default=False,
    ),
    custom_model: Path | None = typer.Option(  # noqa: B008
        None,
        "--custom-model",
        help="Custom trained model folder path to do inference. Needs to contain the same files"
        "as paraplume/trained_models/large which is the output of a training phase. ",
    ),
    name: str = typer.Option(
        "paratope_",
        "--name",
        help="Prefix to add to the file.",
    ),
    gpu: int = typer.Option(
        0,
        "--gpu",
        help="Choose index of GPU device to use if multiple GPUs available. By default it's the"
        "first one (index 0). -1 forces cpu usage. If no GPU is available, CPU is used.",
    ),
    emb_proc_size: int = typer.Option(
        100,
        "--emb-proc-size",
        help="We create embeddings batch by batch to avoid memory explosion. This is the batch"
        "size. Optimal value depends on your computer. Defaults to 100.",
    ),
    result_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--result-folder",
        "-r",
        help="Folder path where to save the results. If not passed the result is saved in the input"
        " data folder.",
    ),
    compute_sequence_embeddings: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--compute-sequence-embeddings",
        help="Compute both paratope and classical sequence embeddings for each sequence "
        "and each of the 6 PLMs AbLang2, Antiberty, ESM, ProtT5, IgT5 and IgBert. "
        "Only possible when using the default trained_models/large.",
    ),
    single_chain: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--single-chain",
        help="Infer paratope on single chain data. Default to False.",
    ),
    large: bool = typer.Option(  # noqa: FBT001
        True,  # noqa: FBT003
        "--large/--small",
        help="Use default Paraplume which uses the 6 PLMs AbLang2,Antiberty,ESM,ProtT5,IgT5 and "
        "IgBert (--large) or the smallest version using only ESM-2 embeddings (--small).",
    ),
    compute_shap: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--compute-shap",
        help="Compute SHAP importance analysis and generate visualizations. A folder 'shap_results'"
        " will be created with a plot inside for each sequence.",
    ),
) -> pd.DataFrame:
    """Predict paratope from sequence."""
    df = pd.read_csv(file_path)

    # Set up SHAP output directory
    shap_output_dir = file_path.parent / Path("shap_results")
    if compute_shap:
        shap_output_dir.mkdir(exist_ok=True)

    predict_paratope(
        df,
        custom_model=custom_model,
        gpu=gpu,
        emb_proc_size=emb_proc_size,
        compute_sequence_embeddings=compute_sequence_embeddings,
        single_chain=single_chain,
        large=large,
        compute_shap=compute_shap,
        shap_output_dir=shap_output_dir,
    )

    if result_path is None:
        result_path = file_path.parents[0] / Path(f"{name}" + file_path.stem)
    else:
        result_path.mkdir(exist_ok=True, parents=True)
        result_path = result_path / Path(f"{name}" + file_path.stem)
    df.to_pickle(result_path.with_suffix(".pkl"))
    return df


def predict_paratope_seq(  # noqa: PLR0913
    sequence_heavy: str = "",
    sequence_light: str = "",
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
    device = get_device(gpu)
    if not custom_model:
        subfolder = "large_1_0" if large else "small_1_0"
        with resources.as_file(
            resources.files("paraplume.trained_models") / subfolder
        ) as model_path:
            custom_model = model_path
    summary_dict_path = custom_model / Path("summary_dict.json")
    with summary_dict_path.open(encoding="utf-8") as f:
        summary_dict = json.load(f)
    input_size = int(summary_dict["input_size"])
    dims = [int(each) for each in summary_dict["dims"].split(",")]
    dropouts = [0] * len(dims)

    model_path = custom_model / Path("checkpoint.pt")
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    nested = any("0.0." in key for key in state_dict.keys())

    model = build_model(input_size, dims, dropouts, nested=nested)
    log.info("Loading model.", path=model_path.as_posix())
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    llm_models = summary_dict["embedding_models"]
    if llm_models == "all":
        llm_models = "ablang2,igbert,igT5,esm,antiberty,prot-t5"
    llm_list = llm_models.split(",")

    seq_emb_list = []
    for llm in llm_list:
        seq_emb_list.append(
            llm_to_func[llm]([sequence_heavy], [sequence_light], gpu=gpu, single_chain=single_chain)
        )
    embeddings = torch.cat(seq_emb_list, dim=-1).to(device)
    output = model(embeddings).cpu().detach().numpy().flatten().tolist()
    heavy, light = len(sequence_heavy), len(sequence_light)
    return output[:heavy], output[heavy : heavy + light]


@app.command()
def seq_to_paratope(
    sequence_heavy: str = typer.Option(
        "",
        "--heavy-chain",
        "-h",
        help="Heavy chain amino acid sequence.",
        show_default=False,
    ),
    sequence_light: str = typer.Option(
        "",
        "--light-chain",
        "-l",
        help="Light chain amino acid sequence.",
        show_default=False,
    ),
    custom_model: Path | None = typer.Option(  # noqa: B008
        None,
        "--custom-model",
        help=(
            "Custom trained model folder path to do inference. Needs to contain the same files "
            "as paraplume/trained_models/large which is the output of a training phase."
        ),
    ),
    gpu: int = typer.Option(
        0,
        "--gpu",
        help="Choose index of GPU device to use if multiple GPUs available. By default it's the "
        "first one (index 0). -1 forces cpu usage. If no GPU is available, CPU is used.",
    ),
    large: bool = typer.Option(  # noqa: FBT001
        True,  # noqa: FBT003
        "--large/--small",
        help="Use default Paraplume which uses the 6 PLMs AbLang2,Antiberty,ESM,ProtT5,IgT5 and "
        "IgBert (--large) or the smallest version using only ESM-2 embeddings (--small).",
    ),
) -> None:
    """Predict paratope from sequence."""
    single_chain = (sequence_heavy is None) or (sequence_light is None)
    output_heavy, output_light = predict_paratope_seq(
        sequence_heavy=sequence_heavy,
        sequence_light=sequence_light,
        custom_model=custom_model,
        gpu=gpu,
        large=large,
        single_chain=single_chain,
    )
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
