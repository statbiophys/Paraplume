import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import typer
from graph_torch_dataset import create_graph_dataloader
from lightning.fabric import Fabric
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import average_precision_score, roc_auc_score

from model import EGNN_Model
from paraplume.utils import get_dim, save_plot

# Typer for CLI interface
app = typer.Typer(add_completion=False)
torch.cuda.empty_cache()


class EGNNLightning(LightningModule):
    def __init__(
        self,
        model: EGNN_Model,
        learning_rate: float,
        criterion: nn.Module,
        positive_weight: float,
        patience: int,
        l2_pen: float = 0.0,
        mask_prob:float=0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.positive_weight = torch.tensor(
            [positive_weight], device=self.device
        )  # Weight for BCE Loss
        self.patience = patience

        # Metrics storage
        self.train_loss_list = []
        self.average_train_loss_list=[]
        self.val_loss_list = []
        self.auc_list = []
        self.ap_list = []
        self.l2_pen = l2_pen

        # To store predictions and labels across batches
        self.predictions = []
        self.targets = []
        self.mask_prob=mask_prob

    def forward(self, feats, coors, edges):
        return self.model(feats, coors, edges)

    def training_step(self, batch, batch_idx):
        (feats, coors, edges), labels, *_ = batch
        drop_mask = torch.rand(feats.size(), device=feats.device) >= self.mask_prob
        feats=feats * drop_mask.float()
        pred = self(feats, coors, edges).squeeze()
        labels = labels.squeeze()
        loss = self.criterion(pred, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.train_loss_list.append(loss.item())  # Append all batch losses

        return loss

    def validation_step(self, batch, batch_idx):
        (feats, coors, edges), labels, *_ = batch
        pred = self(feats, coors, edges).squeeze()
        labels = labels.squeeze()
        loss = self.criterion(pred, labels)

        # Detach and store predictions and labels for later aggregation
        out = torch.sigmoid(pred).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        self.predictions.append(out)
        self.targets.append(labels)

        # Batch-level loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Concatenate predictions and targets from all batches
        all_preds = np.concatenate(self.predictions)
        all_labels = np.concatenate(self.targets)

        # Compute epoch-level metrics
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        self.ap_list.append(ap)
        self.auc_list.append(auc)
        # Log epoch-level metrics
        self.log("epoch_val_auc", auc, prog_bar=True)
        self.log("epoch_val_ap", ap, prog_bar=True)
        self.val_loss_list.append(self.trainer.callback_metrics["val_loss"].item())

        # Clear the stored predictions and targets
        self.predictions = []
        self.targets = []

    def on_train_epoch_end(self):
        avg_train_loss = sum(self.train_loss_list) / len(self.train_loss_list)
        self.average_train_loss_list.append(avg_train_loss)  # Save average train loss per epoch

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_pen
        )
        return optimizer


class EGNNDataModule(LightningDataModule):
    def __init__(
        self,
        train_csv_path: Path,
        val_csv_path: Path,
        train_folder_path: Path,
        val_folder_path: Path,
        embedding_models: str,
        graph_distance: float,
        region: str,
        pdb_folder_path_train: Path,
        pdb_folder_path_val: Path,
        num_workers:int=1
    ):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.train_folder_path = train_folder_path
        self.val_folder_path = val_folder_path
        self.embedding_models = embedding_models
        self.batch_size = 1
        self.graph_distance = graph_distance
        self.region = region
        self.pdb_folder_path_train = pdb_folder_path_train
        self.pdb_folder_path_val = pdb_folder_path_val
        self.num_workers=num_workers

    def setup(self, stage=None):
        # Load datasets
        with open(self.train_folder_path / "dict.json", encoding="utf-8") as f:
            dict_train = json.load(f)
        with open(self.val_folder_path / "dict.json", encoding="utf-8") as f:
            dict_val = json.load(f)
        train_embeddings = torch.load(self.train_folder_path / "embeddings.pt")
        val_embeddings = torch.load(self.val_folder_path / "embeddings.pt")

        train_csv = pd.read_csv(self.train_csv_path)
        val_csv = pd.read_csv(self.val_csv_path)

        self.train_loader = create_graph_dataloader(
            csv=train_csv,
            dataset_dict=dict_train,
            embeddings=train_embeddings,
            shuffle=True,
            pdb_folder_path=self.pdb_folder_path_train,
            embedding_models=self.embedding_models.split(","),
            graph_distance=self.graph_distance,
            region=self.region,
            num_workers=self.num_workers,
        )
        self.val_loader = create_graph_dataloader(
            csv=val_csv,
            dataset_dict=dict_val,
            embeddings=val_embeddings,
            pdb_folder_path=self.pdb_folder_path_val,
            embedding_models=self.embedding_models.split(","),
            graph_distance=self.graph_distance,
            region=self.region,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


@app.command()
def main(
    train_folder_path: Path = typer.Argument(
        ...,
        help="Path of trainfolder.",
        show_default=False,
    ),
    val_folder_path: Path = typer.Argument(
        ...,
        help="Path of valfolder.",
        show_default=False,
    ),
    train_csv_path:Path=typer.Argument(
        ...,
        help="Path of train csv.",
        show_default=False,
    ),
    val_csv_path:Path=typer.Argument(
        ...,
        help="Path of val csv.",
        show_default=False,
    ),
    learning_rate: float = typer.Option(
        0.001, "--lr", help="Learning rate to use for training."
    ),
    n_epochs: int = typer.Option(
        1, "--n_epochs", "-n", help="Number of n_epochs to use for training."
    ),
    result_folder: Path = typer.Option(
        Path("./result/"), "--result_folder", "-r", help="Where to save results."
    ),
    positive_weight: float = typer.Option(
        3, "--pos-weight", help="Weight to give to positive labels."
    ),
    batch_size: int = typer.Option(
        16, "--batch-size", "-bs", help="Batch size. Defaults to 10."
    ),
    override: bool = typer.Option(
        False, "--override", help="Override results. Defaults to False"
    ),
    seed: int = typer.Option(0, "--seed", help="Seed to use for training."),
    pdb_folder_path_val: Path = typer.Option(
        "/home/gathenes/paragraph_benchmark/abb3_pdbs_renumbered",
        "--pdb-folder-path-val",
        help="Pdb folder path.",
    ),
    pdb_folder_path_train: Path = typer.Option(
        "/home/gathenes/all_structures/imgt_renumbered_expanded/",
        "--pdb-folder-path-train",
        help="Pdb folder path.",
    ),
    embedding_models:str=typer.Option(
        "one-hot", "--emb-models", help="Embedding models to use, separated by commas. \
            Models should be in 'ablang2','igbert','igT5','esm','antiberty',prot-t5','all'. \
            All means all of the preceding models. One-hot means the traditional one hot encoding" \
            "of amino acid. Default to one-hot"
    ),
    dropout:float=typer.Option(
        0,"--dropout", help="Dropout for EGNN. Defaults to 0."
    ),
    num_graph_layers:int=typer.Option(
        6, "--num-graph-layers", help="Graph layers dimensions for EGNN. \
        Defaults to 6."),
    linear_layers_dims:str=typer.Option(
        "10,10", "--linear-layers-dims", help="Linear layers dimensions for EGNN. \
            Defaults to 10,10."),
    mask_prob:float=typer.Option(
        0, "--mask-prob", help="Probability with which to mask each embedding coefficient. \
            Defaults to 0"
    ),
    patience:int=typer.Option(
        100, "--patience", help="Patience to use for early stopping. 0 means no early stopping. \
        Defaults to 0."
    ),
    l2_pen:float=typer.Option(
        0, "--l2-pen", help="L2 penalty to use for the model weights."
    ),
    graph_distance:float=typer.Option(
        10, "--graph-distance", help="Distance to use to compute graph"
    ),
    region:str=typer.Option(
        "cdrs","--region", help="Region to use for training. Defaults to cdrs."),
    num_workers:int=typer.Option(
        1, "--num-workers", help="Number of workers for data loading. Defaults to 1."
    ),
    infer_edges: bool = typer.Option(
        False, "--infer-edges", help="Infer edges instead of using sparse graph."
    ),
    gpu:int=typer.Option(
        0, "--gpu", help="Which GPU to use",
    ),
) -> None:
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    nodes = int(os.environ.get("SLURM_NNODES", 1))
    devices = [gpu]

    # read params and share across nodes
    fabric = Fabric(devices=devices, num_nodes=nodes)
    fabric.launch()
    fabric.barrier()
    if seed>0:
        torch.manual_seed(seed)
    if (result_folder / Path("graph_summary_dict.json")).exists() and not override:
        print("Not overriding results.")
        return
    elif (result_folder / Path("summary_dict.json")).exists():
        print("Overriding results.")
    result_folder.mkdir(exist_ok=True, parents=True)

    # Define model
    feature_dim = 22 if embedding_models == "one-hot" else get_dim(embedding_models.split(","))
    if infer_edges:
        edge_dim = 0
    else:
        edge_dim = 1
    model = EGNN_Model(
        num_feats=feature_dim,
        graph_hidden_layer_output_dims=[feature_dim] * num_graph_layers,
        linear_hidden_layer_output_dims=[int(x) for x in linear_layers_dims.split(",")],
        edge_dim=edge_dim,
        dropout=dropout,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]))

    # Define data module
    data_module = EGNNDataModule(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        train_folder_path=train_folder_path,
        val_folder_path=val_folder_path,
        embedding_models=embedding_models,
        graph_distance=graph_distance,
        region=region,
        pdb_folder_path_train=pdb_folder_path_train,
        pdb_folder_path_val=pdb_folder_path_val,
        num_workers=num_workers,
    )

    # Define LightningModule
    egnn_model = EGNNLightning(
        model=model,
        learning_rate=learning_rate,
        criterion=criterion,
        positive_weight=positive_weight,
        patience=patience,
        l2_pen=l2_pen,
        mask_prob=mask_prob,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=result_folder, monitor="epoch_val_ap", mode="max", save_top_k=1
    )
    early_stopping = EarlyStopping(monitor="epoch_val_ap", patience=patience, mode="max")
    logger = CSVLogger(save_dir=str(result_folder), name="logs")

    # Trainer
    trainer = Trainer(
        devices=devices,
        max_epochs=n_epochs,
        precision="16-mixed",
        accelerator="gpu",
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accumulate_grad_batches=batch_size,
        strategy="ddp",

    )

    # Train the model
    trainer.fit(egnn_model, data_module)

    args_dict = {
        "train_folder_path": str(train_folder_path),
        "val_folder_path": str(val_folder_path),
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "result_folder": str(result_folder),
        "positive_weight": positive_weight,
        "batch_size": batch_size,
        "override": override,
        "seed": seed,
        "embedding_models": embedding_models,
        "infer-edges": infer_edges,
        "pdb_folder_path_val": str(pdb_folder_path_val),
        "pdb_folder_path_train": str(pdb_folder_path_train),
        "dropout":dropout,
        "patience":patience,
        "l2_pen": l2_pen,
        "num_graph_layers": num_graph_layers,
        "linear_layers_dims": linear_layers_dims,
        "mask_prob": mask_prob,
        "graph_distance": graph_distance,
        "region": region,
    }
    train_loss_list = egnn_model.average_train_loss_list
    val_loss_list = egnn_model.val_loss_list[1:]
    ap_list = egnn_model.ap_list[1:]
    auc_list = egnn_model.auc_list[1:]
    best_epoch = np.argmax(ap_list)
    args_dict["best_epoch"] = str(best_epoch + 1)
    args_dict["best_ap"] = str(ap_list[best_epoch])
    save_plot_path = result_folder / Path("summary_plot.png")
    save_plot(
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list,
        auc_list=auc_list,
        ap_list=ap_list,
        save_plot_path=save_plot_path,
    )
    with open(result_folder / Path("graph_summary_dict.json"), "w", encoding="utf-8") as json_file:
        json.dump(args_dict, json_file, indent=4)

# Assuming best_checkpoint_path points to the correct checkpoint file
    best_checkpoint_path = checkpoint_callback.best_model_path

    # Define the model architecture
    best_model = EGNN_Model(
        num_feats=feature_dim,
        graph_hidden_layer_output_dims=[feature_dim] * num_graph_layers,
        linear_hidden_layer_output_dims=[int(x) for x in linear_layers_dims.split(",")],
        edge_dim=edge_dim,
        dropout=dropout,
    )

    # Load the checkpoint (weights_only is not a valid argument for torch.load)
    checkpoint = torch.load(best_checkpoint_path, map_location="cpu")  # Adjust device if needed

    # Extract and load the state dictionary
    if "state_dict" in checkpoint:
        # PyTorch Lightning saves the model weights under "state_dict"
        best_model.load_state_dict({k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")})
    else:
        # Direct PyTorch checkpoint
        best_model.load_state_dict(checkpoint)

    # Save the model's state dictionary
    model_save_path = result_folder / Path("graph_checkpoint.pt")
    torch.save(best_model.state_dict(), model_save_path)



if __name__ == "__main__":
    app()
