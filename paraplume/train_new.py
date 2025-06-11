"""Train model."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import typer
from torch import nn
from torch.nn import Dropout, Linear, ReLU, Sequential
from torch.utils.data import DataLoader

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


def get_outputs(
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


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""

    n_epochs: int = 3
    mask_prob: float = 0.0
    patience: int = 0
    gpu: int = 1

    def get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        return torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingMetrics:
    """Data class to store training metrics."""

    train_losses: list[float]
    val_losses: list[float]
    auc_scores: list[float]
    ap_scores: list[float]
    f1_scores: list[float]
    mcc_scores: list[float]
    thresholds: list[float]
    best_val_mcc_thresholds: list[float]
    best_val_f1_thresholds: list[float]

    def to_tuple(self) -> tuple:
        """Convert metrics to tuple for backward compatibility."""
        return (
            self.train_losses,
            self.val_losses,
            self.auc_scores,
            self.ap_scores,
            self.thresholds,
            self.best_val_mcc_thresholds,
            self.best_val_f1_thresholds,
            self.f1_scores,
            self.mcc_scores,
        )


class MultiTaskTrainer:
    """Trainer class for multi-task learning."""

    def __init__(
        self,
        shared_module: nn.Sequential,
        main_task_module: nn.Sequential,
        other_tasks: list[nn.Sequential],
        optimizer: torch.optim.Adam,
        criterion: nn.BCEWithLogitsLoss,
        embedding_models_list: list[str],
        model_save_path: Path,
        config: TrainingConfig,
    ):
        self.shared_module = shared_module
        self.main_task_module = main_task_module
        self.other_tasks = other_tasks
        self.optimizer = optimizer
        self.criterion = criterion
        self.embedding_models_list = embedding_models_list
        self.model_save_path = model_save_path
        self.config = config
        self.device = config.get_device()

        # Initialize aggregator (assuming UPGrad is imported)
        self.aggregator = UPGrad().to(self.device)

        # Initialize metrics storage
        self.metrics = TrainingMetrics(
            train_losses=[],
            val_losses=[],
            auc_scores=[],
            ap_scores=[],
            f1_scores=[],
            mcc_scores=[],
            thresholds=[],
            best_val_mcc_thresholds=[],
            best_val_f1_thresholds=[],
        )

        # Initialize early stopping
        self.early_stopping = (
            EarlyStopping(patience=config.patience, path=model_save_path, best_score=0)
            if config.patience > 0
            else None
        )

    def _prepare_batch_data(self, batch_data: tuple) -> tuple:
        """Prepare and move batch data to device."""
        (
            embedding_raw,
            main_labels_raw,
            len_heavy_raw,
            len_light_raw,
            *other_labels_raw,
        ) = batch_data

        embedding = embedding_raw.to(self.device)
        main_labels = main_labels_raw.to(self.device)
        len_heavy = len_heavy_raw.to(self.device)
        len_light = len_light_raw.to(self.device)
        other_labels = [label.to(self.device) for label in other_labels_raw]

        return embedding, main_labels, len_heavy, len_light, other_labels

    def _process_embeddings(
        self,
        embedding: torch.Tensor,
        main_labels: torch.Tensor,
        len_heavy: torch.Tensor,
        len_light: torch.Tensor,
        other_labels: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Process embeddings and labels for the batch."""
        embedding_list = []
        main_labels_list = []
        other_labels_list = [[] for _ in other_labels]

        for i in range(len_heavy.shape[-1]):
            heavy, light = len_heavy[i], len_light[i]

            # Get embedding
            emb = get_embedding(
                embedding=embedding[i],
                embedding_models=self.embedding_models_list,
                heavy=heavy,
                light=light,
            )

            # Apply masking
            drop_mask = torch.rand(emb.size(), device=emb.device) >= self.config.mask_prob
            emb = emb * drop_mask.float()
            embedding_list.append(emb)

            # Process labels
            lab_main = main_labels[i][: heavy + light]
            main_labels_list.append(lab_main)

            for j, each in enumerate(other_labels):
                other_label = each[i][: heavy + light]
                other_labels_list[j].append(other_label)

        # Concatenate all processed data
        processed_embedding = torch.cat(embedding_list, dim=0)
        processed_main_labels = torch.cat(main_labels_list, dim=0)
        processed_other_labels = [torch.cat(each, dim=0) for each in other_labels_list]

        return processed_embedding, processed_main_labels, processed_other_labels

    def _compute_losses(
        self, features: torch.Tensor, main_labels: torch.Tensor, other_labels: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list]:
        """Compute losses for all tasks."""
        # Main task loss
        output_main = self.main_task_module(features).view(-1)
        loss_main = self.criterion(output_main, main_labels)

        # Other tasks losses
        other_outputs = []
        other_tasks_params = []
        other_losses = []

        for task in self.other_tasks:
            task_on_device = task.to(self.device)
            other_outputs.append(task_on_device(features).view(-1))
            other_tasks_params.append(task_on_device.parameters())

        for other_output, other_label in zip(other_outputs, other_labels, strict=False):
            other_losses.append(self.criterion(other_output, other_label))

        # Combine losses and parameters
        losses = [loss_main] + other_losses
        task_params = [self.main_task_module.parameters()] + other_tasks_params
        losses = [loss.to(self.device) for loss in losses]

        return losses, task_params

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.shared_module.train()
        self.main_task_module.train()

        total_loss = 0.0

        for batch_data in train_loader:
            embedding, main_labels, len_heavy, len_light, other_labels = self._prepare_batch_data(
                batch_data
            )

            # Process embeddings and labels
            (
                processed_embedding,
                processed_main_labels,
                processed_other_labels,
            ) = self._process_embeddings(embedding, main_labels, len_heavy, len_light, other_labels)

            # Forward pass
            self.optimizer.zero_grad()
            self.shared_module = self.shared_module.to(self.device)
            self.main_task_module = self.main_task_module.to(self.device)

            features = self.shared_module(processed_embedding)

            # Compute losses
            losses, task_params = self._compute_losses(
                features, processed_main_labels, processed_other_labels
            )

            # Multi-task learning backward pass
            mtl_backward(
                losses=losses,
                features=features,
                tasks_params=task_params,
                shared_params=self.shared_module.parameters(),
                aggregator=self.aggregator,
            )

            self.optimizer.step()
            total_loss += losses[0].item() * processed_embedding.size(0)

        return total_loss / len(train_loader.dataset)

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        """Validate for one epoch."""
        val_loss = 0.0
        all_outputs = np.array([], dtype=np.float32)
        all_targets = np.array([], dtype=np.float32)

        with torch.no_grad():
            model = nn.Sequential(self.shared_module, self.main_task_module)
            model.eval()

            for embedding_raw, labels_raw, len_heavy, len_light in val_loader:
                embedding, labels = embedding_raw.to(self.device), labels_raw.to(self.device)

                labels, output = get_outputs(
                    embedding=embedding,
                    labels=labels,
                    len_heavy=len_heavy,
                    len_light=len_light,
                    model=model,
                    embedding_models_list=self.embedding_models_list,
                )

                output_sigmoid = torch.sigmoid(output).view(-1).detach().cpu().numpy()
                output = output.view(-1)
                loss = self.criterion(output, labels)
                val_loss += loss.item() * embedding.size(0)

                labels = labels.detach().cpu().numpy()
                all_outputs = np.concatenate((all_outputs, output_sigmoid))
                all_targets = np.concatenate((all_targets, labels))

        val_loss /= len(val_loader.dataset)
        return val_loss, all_outputs, all_targets

    def _update_metrics(
        self, train_loss: float, val_loss: float, outputs: np.ndarray, targets: np.ndarray
    ):
        """Update training metrics."""
        self.metrics.train_losses.append(train_loss)
        self.metrics.val_losses.append(val_loss)

        # Calculate evaluation metrics
        (
            auc,
            ap,
            best_threshold,
            best_validation_mcc_threshold,
            best_validation_f1_threshold,
            max_f1_score,
            max_mcc_score,
        ) = get_metrics(all_outputs=outputs, all_targets=targets)

        self.metrics.auc_scores.append(auc)
        self.metrics.ap_scores.append(ap)
        self.metrics.f1_scores.append(max_f1_score)
        self.metrics.mcc_scores.append(max_mcc_score)
        self.metrics.best_val_mcc_thresholds.append(best_validation_mcc_threshold)
        self.metrics.best_val_f1_thresholds.append(best_validation_f1_threshold)
        self.metrics.thresholds.append(best_threshold)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingMetrics:
        """
        Train the multi-task model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns
        -------
            TrainingMetrics: Object containing all training metrics
        """
        for epoch in range(1, self.config.n_epochs + 1):
            # Training phase
            train_loss = self._train_epoch(train_loader)

            # Validation phase
            val_loss, all_outputs, all_targets = self._validate_epoch(val_loader)

            # Update metrics
            self._update_metrics(train_loss, val_loss, all_outputs, all_targets)

            log.info("Training progress", epoch=epoch, train_loss=train_loss, val_loss=val_loss)

            # Early stopping check
            if self.early_stopping:
                model = nn.Sequential(self.shared_module, self.main_task_module)
                self.early_stopping(-self.metrics.ap_scores[-1], model)
                if self.early_stopping.early_stop:
                    log.info("Early stopping triggered", epoch=epoch)
                    break
            else:
                model = nn.Sequential(self.shared_module, self.main_task_module)
                torch.save(model.state_dict(), self.model_save_path.as_posix())

        return self.metrics


def train(
    shared_module: nn.Sequential,
    main_task_module: nn.Sequential,
    other_tasks: list[nn.Sequential],
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Adam,
    model_save_path: Path,
    criterion: nn.BCEWithLogitsLoss,
    embedding_models_list: list[str],
    n_epochs: int = 3,
    mask_prob: float = 0.0,
    patience: int = 0,
    gpu: int = 1,
) -> tuple:
    """
    Train model given parameters (backward compatibility wrapper).

    This function maintains the same interface as the original for backward compatibility.
    """
    config = TrainingConfig(
        n_epochs=n_epochs,
        mask_prob=mask_prob,
        patience=patience,
        gpu=gpu,
    )

    trainer = MultiTaskTrainer(
        shared_module=shared_module,
        main_task_module=main_task_module,
        other_tasks=other_tasks,
        optimizer=optimizer,
        criterion=criterion,
        embedding_models_list=embedding_models_list,
        model_save_path=model_save_path,
        config=config,
    )

    metrics = trainer.train(train_loader, val_loader)
    return metrics.to_tuple()


def main(
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
    learning_rate: float = typer.Option(0.001, "--lr", help="Learning rate to use for training."),
    n_epochs: int = typer.Option(
        1, "--n_epochs", "-n", help="Number of epochs to use for training."
    ),
    result_folder: Path = typer.Option(  # noqa : B008
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
    override: bool = typer.Option(
        False, "--override", help="Override results. Defaults to False"
    ),
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
        other_tasks = [Sequential(Linear(dims_list[-1], 1)) for _ in alphas_list]
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
