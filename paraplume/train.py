"""Train model."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import typer
from torch import nn
from tqdm import tqdm

from paraplume.torch_dataset import create_dataloader
from paraplume.utils import (
    EarlyStopping,
    build_model,
    get_device,
    get_logger,
    get_metrics,
    save_plot,
)

app = typer.Typer(add_completion=False)
log = get_logger()


def train(  # noqa : PLR0913, PLR0915
    model: torch.nn.Sequential,
    train_loader: torch.utils.data.dataloader.DataLoader,
    val_loader: torch.utils.data.dataloader.DataLoader,
    optimizer: torch.optim.Adam,
    model_save_path: Path,
    criterion: Callable,
    n_epochs: int = 3,
    mask_prob: float = 0,
    patience: int = 0,
    gpu: int = 0,
):
    """Train model given parameters.

    Args:
        shared_module (torch.nn.Sequential): _description_
        main_task_module (torch.nn.Sequential): _description_
        other_tasks (List[torch.nn.Sequential]): _description_
        train_loader (torch.utils.data.dataloader.DataLoader): _description_
        val_loader (torch.utils.data.dataloader.DataLoader): _description_
        optimizer (torch.optim.Adam): _description_
        model_save_path (Path): _description_
        embedding_models_list (List[str]): _description_
        n_epochs (int, optional): _description_. Defaults to 3.
        criterion (_type_, optional): _description_. Defaults to nn.BCELoss().
        mask_prob (float, optional): _description_. Defaults to 0.
        patience (int): Number of epochs after which we stop training.
        Gpu (int): Gpu to use.

    Raises
    ------
        ValueError: _description_

    Returns
    -------
        _type_: _description_
    """
    device = get_device(gpu)
    train_loss_list = []
    val_loss_list = []
    auc_list = []
    ap_list = []
    f1_list = []
    mcc_list = []
    threshold_list = []
    best_val_mcc_threshold_list = []
    best_val_f1_threshold_list = []
    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, path=model_save_path, best_score=0)
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        model.train()
        # Training
        for (
            _,
            ab_emb_cpu,
            label_cpu,
            _,
            _,
            _,
        ) in tqdm(train_loader):
            ab_emb = ab_emb_cpu.to(device)
            label = label_cpu.to(device)
            # Apply dropout (as in original logic)
            drop_mask = torch.rand_like(ab_emb) >= mask_prob
            ab_emb = ab_emb * drop_mask.float()
            optimizer.zero_grad()
            output = model(ab_emb).view(-1)
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * ab_emb.size(0)
        train_loss /= float(len(train_loader.dataset))
        train_loss_list.append(train_loss)
        log.info("Saving train loss and epoch", epoch=epoch, train_loss=train_loss)
        # Validation
        val_loss = 0.0
        all_outputs = np.array([], dtype=np.float32)
        all_targets = np.array([], dtype=np.float32)
        with torch.no_grad():
            model.eval()
            for _, ab_emb_cpu, label_cpu, _, _, _ in val_loader:
                ab_emb = ab_emb_cpu.to(device)
                label = label_cpu.to(device).float()
                output = model(ab_emb).view(-1)
                loss = criterion(output, label)
                output = output.detach().cpu().numpy()
                val_loss += loss.item() * ab_emb.size(0)
                label_np = label.detach().cpu().numpy()
                all_outputs = np.concatenate((all_outputs, output))
                all_targets = np.concatenate((all_targets, label_np))
            val_loss /= float(len(val_loader.dataset))
            val_loss_list.append(val_loss)

        # Get metrics
        (
            auc,
            ap,
            best_threshold,
            best_validation_mcc_threshold,
            best_validation_f1_threshold,
            max_f1_score,
            max_mcc_score,
        ) = get_metrics(all_outputs=all_outputs, all_targets=all_targets)
        auc_list.append(auc)
        ap_list.append(ap)
        f1_list.append(max_f1_score)
        mcc_list.append(max_mcc_score)
        best_val_mcc_threshold_list.append(best_validation_mcc_threshold)
        best_val_f1_threshold_list.append(best_validation_f1_threshold)
        threshold_list.append(best_threshold)

        # Early stopping
        if patience > 0:
            early_stopping(-ap, model)
            if early_stopping.early_stop:
                log.info("Early stopping, last epoch", epoch=epoch)
                break
        else:
            torch.save(model.state_dict(), model_save_path.as_posix())
    return (
        train_loss_list,
        val_loss_list,
        auc_list,
        ap_list,
        threshold_list,
        best_val_mcc_threshold_list,
        best_val_f1_threshold_list,
        f1_list,
        mcc_list,
    )


@app.command()
def main(  # noqa: PLR0913
    train_folder_path: Path = typer.Argument(  # noqa : B008
        ...,
        help="Path of trainfolder.",
        show_default=False,
    ),
    val_folder_path: Path = typer.Argument(  # noqa : B008
        ...,
        help="Path of valfolder.",
        show_default=False,
    ),
    learning_rate: float = typer.Option(0.00001, "--lr", help="Learning rate to use for training."),
    n_epochs: int = typer.Option(
        1, "--n_epochs", "-n", help="Number of epochs to use for training."
    ),
    result_folder: Path = typer.Option(  # noqa : B008
        Path("./result/"), "--result-folder", "-r", help="Where to save results."
    ),
    batch_size: int = typer.Option(4096, "--batch-size", "-bs", help="Batch size."),
    mask_prob: float = typer.Option(
        0.4,
        "--mask-prob",
        help="Probability with which to mask each embedding coefficient.",
    ),
    dropouts: str = typer.Option(
        "0.4,0.4,0.4",
        "--dropouts",
        help="Dropout probabilities for each hidden layer, separated by commas. Example '0.3,0.3'.",
    ),
    dims: str = typer.Option(
        "2000,1000,500",
        "--dims",
        help="Dimensions of hidden layers. Separated by commas. Example '100,100'",
    ),
    override: bool = typer.Option(False, "--override", help="Override results. Defaults to False"),  # noqa : FBT001, FBT003
    seed: int = typer.Option(0, "--seed", help="Seed to use for training."),
    l2_pen: float = typer.Option(
        0.00001, "--l2-pen", help="L2 penalty to use for the model weights."
    ),
    patience: int = typer.Option(
        10,
        "--patience",
        help="Patience to use for early stopping. 0 means no early stopping.",
    ),
    embedding_models: str = typer.Option(
        "all",
        "--emb-models",
        help=(
            "LLM embedding models to use, separated by commas. "
            "LLMs should be in 'ablang2','igbert','igT5','esm','antiberty',prot-t5','all'. "
            "Example 'igT5,esm'."
        ),
    ),
    gpu: int = typer.Option(
        0,
        "--gpu",
        help="Choose index of GPU device to use if multiple GPUs available. By default it's the"
        "first one (index 0). -1 forces cpu usage. If no GPU is available, CPU is used.",
    ),
) -> None:
    """Train the model given provided parameters and data."""
    if (result_folder / Path("summary_dict.json")).exists() and not override:
        print((result_folder / Path("summary_dict.json")).as_posix())
        log.info("Not overriding results.")
        return
    if (result_folder / Path("summary_dict.json")).exists():
        log.info("Overriding results.")
    log.info("CREATING RESULT FOLDER", result_folder=result_folder.as_posix())
    result_folder.mkdir(exist_ok=True, parents=True)
    log.info("PROCESSING ARGUMENTS", arguments="model_list,alphas,seed,dropouts,dims")
    if embedding_models == "all":
        embedding_models = "ablang2,igbert,igT5,esm,antiberty,prot-t5"
    embedding_models_list = embedding_models.split(",")
    train_embeddings = torch.cat(
        [
            torch.load(train_folder_path / Path(f"{model}_embeddings.pt"), weights_only=True)
            for model in embedding_models_list
        ],
        dim=-1,
    )
    val_embeddings = torch.cat(
        [
            torch.load(val_folder_path / Path(f"{model}_embeddings.pt"), weights_only=True)
            for model in embedding_models_list
        ],
        dim=-1,
    )
    if seed > 0:
        torch.manual_seed(seed)
    input_size = train_embeddings.shape[-1]
    dims_list = [int(each) for each in dims.split(",")]
    dropouts_list = [float(each) for each in dropouts.split(",")]
    log.info("LOADING DICTIONARY AND EMBEDDINGS")
    with (train_folder_path / Path("dict.json")).open(encoding="utf-8") as f:
        dict_train = json.load(f)
    with (val_folder_path / Path("dict.json")).open(encoding="utf-8") as f:
        dict_val = json.load(f)
    log.info("CREATING DATALOADER")
    train_loader = create_dataloader(
        dataset_dict=dict_train,
        embeddings=train_embeddings,
        batch_size=batch_size,
    )
    val_loader = create_dataloader(
        dataset_dict=dict_val,
        embeddings=val_embeddings,
        batch_size=batch_size,
    )
    log.info("INITIALIZE MODEL", hidden_layer_dimensions=dims_list, dropouts=dropouts)
    model = build_model(input_size=input_size, dims_list=dims_list, dropouts_list=dropouts_list)
    params = model.parameters()

    criterion = nn.BCELoss()
    log.info("INITIALIZE OPTIMIZER", learning_rate=learning_rate, weight_decay=l2_pen)
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=l2_pen)
    model_save_path = result_folder / Path("checkpoint.pt")
    log.info("TRAIN MODEL", epochs=n_epochs, patience=patience)
    (
        train_loss_list,
        val_loss_list,
        auc_list,
        ap_list,
        threshold_list,
        best_val_mcc_threshold_list,
        best_val_f1_threshold_list,
        f1_list,
        mcc_list,
    ) = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=n_epochs,
        criterion=criterion,
        model_save_path=model_save_path,
        mask_prob=mask_prob,
        patience=patience,
        gpu=gpu,
    )
    log.info(
        "SAVE TRAIN AND VALID METRIC PLOTS",
        path=(result_folder / Path("summary_plot.png")).as_posix(),
    )
    save_plot_path = result_folder / Path("summary_plot.png")
    save_plot(
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list,
        auc_list=auc_list,
        ap_list=ap_list,
        save_plot_path=save_plot_path,
    )
    log.info(
        "SAVE ALL METRICS AND PARAMETERS USED",
        path=(result_folder / Path("summary_dict.json")).as_posix(),
    )
    best_epoch = np.argmax(ap_list)
    args_dict = {
        "chain": "paired",
        "train_folder_path": str(train_folder_path),
        "val_folder_path": str(val_folder_path),
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "result_folder": str(result_folder),
        "dims": dims,
        "mask_prob": mask_prob,
        "dropouts": dropouts,
        "batch_size": batch_size,
        "override": override,
        "seed": seed,
        "patience": patience,
        "embedding_models": embedding_models,
        "input_size": str(input_size),
        "best_epoch": str(best_epoch),
        "best_ap_all_res": str(ap_list[best_epoch]),
        "best_auc_all_res": str(auc_list[best_epoch]),
        "threshold_youden": str(threshold_list[best_epoch]),
        "best_val_mcc_threshold": str(best_val_mcc_threshold_list[best_epoch]),
        "best_val_f1_threshold": str(best_val_f1_threshold_list[best_epoch]),
        "f1": str(f1_list[best_epoch]),
        "mcc": str(mcc_list[best_epoch]),
    }
    with (result_folder / Path("summary_dict.json")).open("w", encoding="utf-8") as json_file:
        json.dump(args_dict, json_file, indent=4)


if __name__ == "__main__":
    app()
