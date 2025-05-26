"""Train model."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import typer
from torch import nn
from torch.nn import Dropout, Linear, ReLU, Sequential
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.nn import Module  # used only for typing

from torchjd import mtl_backward
from torchjd.aggregation import UPGrad

from paraplume.torch_dataset import create_dataloader
from paraplume.utils import (
    EarlyStopping,
    get_dim,
    get_embedding,
    get_logger,
    get_metrics,
    save_plot,
)

app = typer.Typer(add_completion=False)
log = get_logger()


def get_outputs( # noqa : PLR0913
    embedding: torch.Tensor,
    labels: torch.Tensor,
    len_heavy: torch.Tensor,
    len_light: torch.Tensor,
    model: torch.nn.Sequential,
    embedding_models_list: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get model outputs and labels.

    Args:
        embedding (torch.Tensor):
        labels (torch.Tensor):
        len_heavy (torch.Tensor): Lenghts of the heavy sequences for each element of the batch.
        len_light (torch.Tensor): Lenghts of the light sequences for each element of the batch.
        model (torch.nn.Sequential):
        embedding_models_list (List): List of embedding models for which to get the\
            pre-computed embeddings.

    Returns
    -------
        Tuple[torch.Tensor,torch.Tensor]: Labels and outputs.
    """
    embedding_list = []
    label_list = []
    for i in range(len_heavy.shape[-1]):
        heavy, light = len_heavy[i], len_light[i]
        emb = get_embedding(
            embedding=embedding[i],
            embedding_models=embedding_models_list,
            heavy=heavy,
            light=light,
        )
        embedding_list.append(emb)
        label_list.append(labels[i][: heavy + light])
    embedding = torch.cat(embedding_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    output = model(embedding)
    return labels, output


def train(  # noqa : PLR0913, PLR0915
    shared_module: torch.nn.Sequential,
    main_task_module: torch.nn.Sequential,
    other_tasks: list[torch.nn.Sequential],
    train_loader: torch.utils.data.dataloader.DataLoader,
    val_loader: torch.utils.data.dataloader.DataLoader,
    optimizer: torch.optim.Adam,
    model_save_path: Path,
    criterion: nn.BCEWithLogitsLoss,
    embedding_models_list: list[str],
    n_epochs: int = 3,
    mask_prob: float = 0,
    patience: int = 0,
    gpu: int = 1,
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

    Raises
    ------
        ValueError: _description_

    Returns
    -------
        _type_: _description_
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    aggregator = UPGrad().to(device)

    train_loss_list = []
    val_loss_list = []
    auc_list = []
    ap_list = []
    f1_list = []
    mcc_list = []
    threshold_list = []
    best_val_mcc_threshold_list = []
    best_val_f1_threshold_list = []

    early_stopping = EarlyStopping(patience=patience, path=model_save_path, best_score=0)
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        shared_module.train()
        main_task_module.train()
        # Training
        for (
            embedding_raw,
            main_labels_raw,
            len_heavy_raw,
            len_light_raw,
            *other_labels_raw,
        ) in tqdm(train_loader):
            embedding = embedding_raw.to(device)
            main_labels = main_labels_raw.to(device)
            len_heavy = len_heavy_raw.to(device)
            len_light = len_light_raw.to(device)
            other_labels = [label.to(device) for label in other_labels_raw]

            embedding_list = []
            main_labels_list = []
            other_labels_list: list[list[torch.Tensor]] = []
            other_labels_list = [[] for _ in other_labels]
            for i in range(len_heavy.shape[-1]):
                heavy, light = len_heavy[i], len_light[i]
                emb = get_embedding(
                    embedding=embedding[i],
                    embedding_models=embedding_models_list,
                    heavy=heavy,
                    light=light,
                )
                lab_main = main_labels[i][: heavy + light]
                main_labels_list.append(lab_main)
                for j, each in enumerate(other_labels):
                    other_label = each[i][: heavy + light]
                    other_labels_list[j].append(other_label)
                drop_mask = torch.rand(emb.size(), device=emb.device) >= mask_prob
                emb = emb * drop_mask.float()
                embedding_list.append(emb)
            embedding = torch.cat(embedding_list, dim=0)
            main_labels = torch.cat(main_labels_list, dim=0)
            other_labels = [torch.cat(each, dim=0) for each in other_labels_list]
            optimizer.zero_grad()
            shared_module = shared_module.to(device)
            main_task_module = main_task_module.to(device)
            features = shared_module(embedding)
            output_main = main_task_module(features).view(-1)
            loss_main = criterion(output_main, main_labels)
            other_outputs = []
            other_tasks_params = []
            for task in other_tasks:
                task_on_device = task.to(device)
                other_outputs.append(task_on_device(features).view(-1))
                other_tasks_params+=[task_on_device.parameters()]
            other_losses = []
            for other_output, other_label in zip(other_outputs, other_labels, strict=False):
                other_losses.append(criterion(other_output, other_label))
            losses = [loss_main]
            task_params = [main_task_module.parameters()]
            losses += [*other_losses]
            task_params += [*other_tasks_params]
            losses = [loss.to(device) for loss in losses]
            mtl_backward(
                losses=losses,
                features=features,
                tasks_params=task_params,
                shared_params=shared_module.parameters(),
                aggregator=aggregator,
            )
            optimizer.step()
            train_loss += loss_main.item() * embedding.size(0)
        train_loss /= float(len(train_loader.dataset))
        train_loss_list.append(train_loss)
        log.info("Saving train loss and epoch", epoch=epoch, train_loss=train_loss)
        # Validation
        val_loss = 0.0
        all_outputs = np.array([], dtype=np.float32)
        all_targets = np.array([], dtype=np.float32)
        with torch.no_grad():
            model = nn.Sequential(shared_module, main_task_module)
            model.eval()
            for embedding_raw, labels_raw, len_heavy, len_light in val_loader:
                embedding, labels = embedding_raw.to(device), labels_raw.to(device)
                labels, output = get_outputs(
                    embedding=embedding,
                    labels=labels,
                    len_heavy=len_heavy,
                    len_light=len_light,
                    model=model,
                    embedding_models_list=embedding_models_list,
                )
                output_sigmoid = torch.sigmoid(output).view(-1).detach().cpu().numpy()
                output = output.view(-1)
                loss = criterion(output, labels)
                val_loss += loss.item() * embedding.size(0)
                labels = labels.detach().cpu().numpy()
                all_outputs = np.concatenate((all_outputs, output_sigmoid))
                all_targets = np.concatenate((all_targets, labels))
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
def main( # noqa : PLR0913, PLR0915
    train_folder_path: Path = typer.Argument( # noqa : B008
        ...,
        help="Path of trainfolder.",
        show_default=False,
    ),
    val_folder_path: Path = typer.Argument( # noqa : B008
        ...,
        help="Path of valfolder.",
        show_default=False,
    ),
    learning_rate: float = typer.Option(0.001, "--lr", help="Learning rate to use for training."),
    n_epochs: int = typer.Option(
        1, "--n_epochs", "-n", help="Number of epochs to use for training."
    ),
    result_folder: Path = typer.Option( # noqa : B008
        Path("./result/"), "--result_folder", "-r", help="Where to save results."
    ),
    positive_weight: float = typer.Option(
        1, "--pos-weight", help="Weight to give to positive labels."
    ),
    batch_size: int = typer.Option(10, "--batch-size", "-bs", help="Batch size. Defaults to 10."),
    mask_prob: float = typer.Option(
        0,
        "--mask-prob",
        help="Probability with which to mask each embedding coefficient. \
            Defaults to 0",
    ),
    dropouts: str = typer.Option("0", "--dropouts", help="Dropout probability. Defaults to 0."),
    dims: str = typer.Option(
        "1000", "--dims", help="Dimension of hidden layers. Separated by commas."
    ),
    override: bool = typer.Option(False, "--override", help="Override results. Defaults to False"), # noqa : FBT001, FBT003
    seed: int = typer.Option(0, "--seed", help="Seed to use for training."),
    l2_pen: float = typer.Option(0, "--l2-pen", help="L2 penalty to use for the model weights."),
    alphas: str = typer.Option(
        "-",
        "--alphas",
        help="Whether to use different alphas labels to help main label. \
        Defaults to empty.",
    ),
    patience: int = typer.Option(
        0,
        "--patience",
        help="Patience to use for early stopping. 0 means no early stopping. \
        Defaults to 0.",
    ),
    embedding_models: str = typer.Option(
        "all",
        "--emb-models",
        help="Embedding models to use, separated by commas. \
            Models should be in 'ablang2','igbert','igT5','esm','antiberty',prot-t5','all'. \
            Default to 'all'.",
    ),
    gpu: int = typer.Option(1, "--gpu", help="Which GPU to use."),
) -> None:
    """Train the model given provided parameters and data."""
    if (result_folder / Path("summary_dict.json")).exists() and not override:
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
    alphas_list = None
    if alphas != "-":
        alphas_list = alphas.split(",")
    if seed > 0:
        torch.manual_seed(seed)
    input_size = get_dim(embedding_models=embedding_models_list)
    dims_list = [int(each) for each in dims.split(",")]
    dropouts_list = [float(each) for each in dropouts.split(",")]
    log.info("LOADING DICTIONARY AND EMBEDDINGS")
    with (train_folder_path / Path("dict.json")).open(encoding="utf-8") as f:
        dict_train = json.load(f)
    train_embeddings = torch.load(train_folder_path / Path("embeddings.pt"), weights_only=True)
    with (val_folder_path / Path("dict.json")).open(encoding="utf-8") as f:
        dict_val = json.load(f)
    val_embeddings = torch.load(val_folder_path / Path("embeddings.pt"), weights_only=True)
    log.info("CREATING DATALOADER")
    train_loader = create_dataloader(
        dataset_dict=dict_train,
        embeddings=train_embeddings,
        batch_size=batch_size,
        alphas=alphas_list,
        mode="train",
    )
    val_loader = create_dataloader(
        dataset_dict=dict_val,
        embeddings=val_embeddings,
        batch_size=batch_size,
        mode="test",
    )
    log.info("INITIALIZE MODEL", hidden_layer_dimensions=dims_list, dropouts=dropouts)
    layers: list[Module] = []
    for i, _ in enumerate(dims_list):
        if i == 0:
            layers.append(Linear(input_size, dims_list[i]))
            layers.append(Dropout(dropouts_list[i]))
            layers.append(ReLU())
        else:
            layers.append(Linear(dims_list[i - 1], dims_list[i]))
            layers.append(Dropout(dropouts_list[i]))
            layers.append(ReLU())
    shared_module = Sequential(*layers)
    main_task_module = Sequential(Linear(dims_list[-1], 1))
    params = [*shared_module.parameters(), *main_task_module.parameters()]
    other_tasks = []
    if alphas_list:
        other_tasks = [
            Sequential(Linear(dims_list[-1], 1))
            for _ in alphas_list
        ]
        for task in other_tasks:
            params += [*task.parameters()]
    log.info(
        "INITIALIZE LOSS FUNCTION",
        weight=positive_weight,
        criterion="BCE with logits loss",
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_weight],
            device=torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"),
        )
    )
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
        shared_module=shared_module,
        main_task_module=main_task_module,
        other_tasks=other_tasks,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=n_epochs,
        criterion=criterion,
        model_save_path=model_save_path,
        mask_prob=mask_prob,
        patience=patience,
        embedding_models_list=embedding_models_list,
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
        "positive_weight": positive_weight,
        "dims": dims,
        "mask_prob": mask_prob,
        "dropouts": dropouts,
        "batch_size": batch_size,
        "override": override,
        "seed": seed,
        "alphas": alphas,
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
