import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import typer
from models import EarlyStopping
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential, Sigmoid
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch_dataset import ParatopeDataset, ParatopeMultiObjectiveDataset
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad
from tqdm import tqdm
from utils import save_plot

app = typer.Typer(add_completion=False)

def create_dataloader(dataset_dict:Dict,residue_embeddings:torch.Tensor, batch_size=10, shuffle:bool=False, alpha:str="4.5", multiobjective=False)->torch.utils.data.dataloader.DataLoader:
    """Take dataset_dict and embeddings and return dataloader.

    Args:
        dataset_dict (Dict): _description_
        residue_embeddings (torch.Tensor): _description_
        batch_size (int, optional): _description_. Defaults to 10.

    Returns:
        torch.utils.data.dataloader.DataLoader: Dataloader to use for training.
    """
    if multiobjective:
        dataset=ParatopeMultiObjectiveDataset(dataset_dict=dataset_dict, residue_embeddings=residue_embeddings, alpha=alpha)
    else:
        dataset = ParatopeDataset(dataset_dict=dataset_dict, residue_embeddings=residue_embeddings, alpha=alpha)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader

def train_multiobjective(
    shared_module,
    task1_module,
    task2_module,
    task3_module,
    task4_module,
    train_loader,
    val_loader,
    optimizer,
    n_epochs=3,
    model_save_path=Path("/home/gathenes/all_structures/data_high_qual/checkpoint.pt"),
    criterion=nn.BCELoss(),
    mask_prob:float=0,
    lr_strategy:Optional[str]=None,  # Can be "step", "plateau", or "cosine"
    big_embedding=False,
):
    A=UPGrad()
    device = torch.device("cpu")
    train_loss_list = []
    val_loss_list = []
    auc_list = []
    ap_list = []

    early_stopping = EarlyStopping(patience=7, path=model_save_path, best_score=0)
    if lr_strategy == "step":
        scheduler = StepLR(optimizer, step_size=40)
    elif lr_strategy == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=4)
    elif lr_strategy == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=25)
    elif lr_strategy:
        raise ValueError(f"Unknown scheduler type: {lr_strategy}")
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        shared_module.train()
        for i, (embedding, len_heavy, len_light, labels1, labels2, labels3, labels4) in enumerate(tqdm(train_loader)):
            embedding, labels1, labels2, labels3 = embedding.to(device), labels1.to(device), labels2.to(device), labels3.to(device)
            embedding_list=[]
            label1_list = []
            label2_list = []
            label3_list = []
            label4_list = []
            for i in range(len_heavy.shape[-1]):
                heavy, light = len_heavy[i], len_light[i]
                ran = list(range(1,heavy+1))+list(range(heavy+2, heavy+light+2))
                emb=embedding[i][ran,:2048]
                if big_embedding:
                    ran2 = list(range(1,heavy+1))+list(range(heavy+4, heavy+light+4))
                    emb2=embedding[i][ran2,2048:]
                    emb=torch.cat([emb,emb2], dim=1)
                lab1=labels1[i][:heavy+light]
                lab2=labels2[i][:heavy+light]
                lab3=labels3[i][:heavy+light]
                lab4=labels4[i][:heavy+light]
                if random.random()<mask_prob:
                    drop_mask = torch.rand(emb.size(), device=emb.device) >= mask_prob
                    emb=emb * drop_mask.float()
                embedding_list.append(emb)
                label1_list.append(lab1)
                label2_list.append(lab2)
                label3_list.append(lab3)
                label4_list.append(lab4)
            embedding = torch.cat(embedding_list, dim=0)
            labels1 = torch.cat(label1_list, dim=0)
            labels2 = torch.cat(label2_list, dim=0)
            labels3 = torch.cat(label3_list, dim=0)
            labels4 = torch.cat(label4_list, dim=0)

            optimizer.zero_grad()
            features = shared_module(embedding)
            output1 = task1_module(features).view(-1)
            output2 = task2_module(features).view(-1)
            output3 = task3_module(features).view(-1)
            output4 = task4_module(features).view(-1)
            loss1 = criterion(output1, labels1)
            loss2 = criterion(output2, labels2)
            loss3 = criterion(output3, labels3)
            loss4 = criterion(output4, labels4)
            mtl_backward(
                losses=[loss1, loss2, loss3, loss4],
                features=features,
                tasks_params=[task1_module.parameters(), task2_module.parameters(),task3_module.parameters(),task4_module.parameters()],
                shared_params=shared_module.parameters(),
                A=A,
            )
            optimizer.step()
            train_loss += loss2.item() * embedding.size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)
        val_loss = 0
        all_outputs = []
        all_targets = []
        print(f"Saving train loss and epoch, epoch={epoch}, train_loss={train_loss}")
        with torch.no_grad():
            model = nn.Sequential(
            shared_module,
            task2_module
            )
            model.eval()
            for embedding, labels, len_heavy, len_light in val_loader:
                embedding, labels = embedding.to(device), labels.to(device)
                embedding_list=[]
                label_list = []
                for i in range(len_heavy.shape[-1]):
                    heavy, light = len_heavy[i], len_light[i]
                    ran = list(range(1,heavy+1))+list(range(heavy+2, heavy+light+2))
                    emb=embedding[i][ran,:2048]
                    if big_embedding :
                        ran2 = list(range(1,heavy+1))+list(range(heavy+4, heavy+light+4))
                        emb2=embedding[i][ran2,2048:]
                        emb=torch.cat([emb,emb2], dim=1)
                    embedding_list.append(emb)
                    label_list.append(labels[i][:heavy+light])
                embedding = torch.cat(embedding_list, dim=0)
                labels = torch.cat(label_list, dim=0)
                output = model(embedding)
                output=output.view(-1)
                loss = criterion(output, labels)
                val_loss += loss.item() * embedding.size(0)
                output = output.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                all_outputs.extend(output)
                all_targets.extend(labels)

            val_loss /= len(val_loader.dataset)
            val_loss_list.append(val_loss)

        # Converting lists to numpy arrays for AUC and ROC calculation
        all_outputs = np.array(all_outputs)
        all_targets = np.array(all_targets)

        # Calculate the AUC score
        auc = roc_auc_score(all_targets, all_outputs)
        auc_list.append(auc)

        ap = average_precision_score(all_targets, all_outputs)
        ap_list.append(ap)

        # Calculate the Average Precision score

        early_stopping(-ap, model)
        if early_stopping.early_stop:
            print(f"Early stopping, last epoch = {epoch}")
            break
        if lr_strategy in ["step","cosine"]:
            scheduler.step()  # StepLR scheduler: decrease lr after `step_size` epochs
        elif lr_strategy == "plateau":
            scheduler.step(-ap)
    return train_loss_list, val_loss_list, auc_list, ap_list


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    n_epochs=3,
    model_save_path=Path("/home/gathenes/all_structures/data_high_qual/checkpoint.pt"),
    criterion=nn.BCELoss(),
    mask_prob:float=0,
    lr_strategy:Optional[str]=None,  # Can be "step", "plateau", or "cosine"
    big_embedding=False,
):
    # Training loop
    device = torch.device("cpu")
    train_loss_list = []
    val_loss_list = []
    auc_list = []
    ap_list = []

    early_stopping = EarlyStopping(patience=7, path=model_save_path, best_score=0)
    if lr_strategy == "step":
        scheduler = StepLR(optimizer, step_size=40)
    elif lr_strategy == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=4)
    elif lr_strategy == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=25)
    elif lr_strategy:
        raise ValueError(f"Unknown scheduler type: {lr_strategy}")
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        model.train()
        for i, (embedding, labels, len_heavy, len_light) in enumerate(tqdm(train_loader)):
            embedding, labels = embedding.to(device), labels.to(device)
            embedding_list=[]
            label_list = []
            for i in range(len_heavy.shape[-1]):
                heavy, light = len_heavy[i], len_light[i]
                ran = list(range(1,heavy+1))+list(range(heavy+2, heavy+light+2))
                emb=embedding[i][ran,:2048]
                if big_embedding:
                    ran2 = list(range(1,heavy+1))+list(range(heavy+4, heavy+light+4))
                    emb2=embedding[i][ran2,2048:]
                    emb=torch.cat([emb,emb2], dim=1)
                lab=labels[i][:heavy+light]
                if random.random()<mask_prob:
                    drop_mask = torch.rand(emb.size(), device=emb.device) >= mask_prob
                    emb=emb * drop_mask.float()
                embedding_list.append(emb)
                label_list.append(lab)
            embedding = torch.cat(embedding_list, dim=0)
            labels = torch.cat(label_list, dim=0)
            optimizer.zero_grad()
            output = model(embedding)
            output=output.view(-1)
            #print(embedding.shape, labels.shape, output.shape)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * embedding.size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)
        val_loss = 0
        all_outputs = []
        all_targets = []
        print(f"Saving train loss and epoch, epoch={epoch}, train_loss={train_loss}")
        if len(val_loader) == 0:
            continue
        with torch.no_grad():
            model.eval()
            for embedding, labels, len_heavy, len_light in val_loader:
                embedding, labels = embedding.to(device), labels.to(device)
                embedding_list=[]
                label_list = []
                for i in range(len_heavy.shape[-1]):
                    heavy, light = len_heavy[i], len_light[i]
                    ran = list(range(1,heavy+1))+list(range(heavy+2, heavy+light+2))
                    emb=embedding[i][ran,:2048]
                    if big_embedding :
                        ran2 = list(range(1,heavy+1))+list(range(heavy+4, heavy+light+4))
                        emb2=embedding[i][ran2,2048:]
                        emb=torch.cat([emb,emb2], dim=1)
                    embedding_list.append(emb)
                    label_list.append(labels[i][:heavy+light])
                embedding = torch.cat(embedding_list, dim=0)
                labels = torch.cat(label_list, dim=0)
                output = model(embedding)
                output=output.view(-1)
                loss = criterion(output, labels)
                val_loss += loss.item() * embedding.size(0)
                output = output.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                all_outputs.extend(output)
                all_targets.extend(labels)

            val_loss /= len(val_loader.dataset)
            val_loss_list.append(val_loss)

        # Converting lists to numpy arrays for AUC and ROC calculation
        all_outputs = np.array(all_outputs)
        all_targets = np.array(all_targets)

        # Calculate the AUC score
        auc = roc_auc_score(all_targets, all_outputs)
        auc_list.append(auc)

        ap = average_precision_score(all_targets, all_outputs)
        ap_list.append(ap)

        # Calculate the Average Precision score
        early_stopping(-ap, model)
        if early_stopping.early_stop:
            print(f"Early stopping, last epoch = {epoch}")
            break
        if lr_strategy in ["step","cosine"]:
            scheduler.step()  # StepLR scheduler: decrease lr after `step_size` epochs
        elif lr_strategy == "plateau":
            scheduler.step(-ap)
    return train_loss_list, val_loss_list, auc_list, ap_list

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
    model_name: str = typer.Option(
        "MLP", "--model", "-m", help="Model to use for training."
    ),
    learning_rate: float = typer.Option(
        0.001, "--lr", help="Learning rate to use for training."
    ),
    n_epochs: int = typer.Option(
        1, "--n_epochs", "-n", help="Number of epochs to use for training."
    ),
    result_folder: Path = typer.Option(
        Path("./result/"), "--result_folder", "-r", help="Where to save results."
    ),
    positive_weight: float = typer.Option(
        1, "--pos-weight", help="Weight to give to positive labels."
    ),
    batch_size:int=typer.Option(
        10, "--batch-size", "-bs", help="Batch size. Defaults to 10."
    ),
    lr_strategy:Optional[str]=typer.Option(
        None, "--lr-strat", help="Learning rate strategy to use. Must be 'step', 'plateau' \
            or 'cosine'. Defaults to None."
    ),
    mask_prob:float=typer.Option(
        0, "--mask-prob", help="Probability with which to mask each embedding coefficient. \
            Defaults to 0"
    ),
    dropouts:str=typer.Option(
        "0", "--dropouts", help="Dropout probability. Defaults to 0."
    ),
    dims:str=typer.Option(
        "1000", "--dims", help="Dimension of hidden layers. Separated by commas."
    ),
    batchnorm:bool=typer.Option(
        False, "--batch-norm", help="Whether to use batchnorm or not. Defaults to False."
    ),
    override:bool=typer.Option(
        False, "--override", help="Override results. Defaults to False"
    ),
    alpha:str=typer.Option(
        4.5, "--alpha", help="Alpha distance to use for labels. Default to 4.5."
    ),
    seed:int=typer.Option(
        0, "--seed", help="Seed to use for training."
    ),
    l2_pen:float=typer.Option(
        0, "--l2-pen", help=("L2 penalty to use for the model weights."
        )
    ),
    multiobjective:bool=typer.Option(
        False,"--multiobjective", help=("Whether to use multiobjective or not.")
    ),
    big_embedding:bool=typer.Option(
        False,"--bigembedding", help=("Whether to use big embeddings or not.")
    ),
) -> None:
    if (result_folder/Path("summary_dict.json")).exists() and not override :
        print("Not overriding results.")
        return
    elif (result_folder/Path("summary_dict.json")).exists():
        print("Overriding results.")
    result_folder.mkdir(exist_ok=True, parents=True)
    args_dict = {
        "train_folder_path": str(train_folder_path),
        "val_folder_path": str(val_folder_path),
        "model_name": model_name,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "result_folder": str(result_folder),
        "positive_weight": positive_weight,
        "dims":dims,
        "mask_prob":mask_prob,
        "dropouts":dropouts,
        "lr_strategy":lr_strategy,
        "batch_size":batch_size,
        "batchnorm":batchnorm,
        "override":override,
        "alpha":alpha,
        "seed":seed,
        "multiobjective":multiobjective,
        "big_embedding":big_embedding,
    }
    if seed>0:
        torch.manual_seed(seed)
    with open(train_folder_path / Path("dict.json")) as f :
        dict_train = json.load(f)
    train_embeddings = torch.load(train_folder_path / Path("embeddings.pt"), weights_only=True)

    with open(val_folder_path / Path("dict.json")) as f :
        dict_val = json.load(f)
    val_embeddings = torch.load(val_folder_path / Path("embeddings.pt"), weights_only=True)

    train_loader = create_dataloader(dataset_dict=dict_train, residue_embeddings=train_embeddings, batch_size=batch_size, shuffle=True, alpha=alpha, multiobjective=multiobjective)
    val_loader = create_dataloader(dataset_dict=dict_val, residue_embeddings=val_embeddings, batch_size=batch_size, alpha=alpha)
    if big_embedding :
        input_size = 2528
    else:
        input_size = 2048
    dims=dims.split(",")
    dims=[int(each) for each in dims]
    dropouts=dropouts.split(",")
    dropouts=[float(each) for each in dropouts]
    layers=[]
    for i, _ in enumerate(dims):
        if i==0:
            layers.append(Linear(input_size, dims[i]))
            layers.append(Dropout(dropouts[i]))
            if batchnorm:
                layers.append(BatchNorm1d)
            layers.append(ReLU())
        else :
            layers.append(Linear(dims[i-1], dims[i]))
            layers.append(Dropout(dropouts[i]))
            if batchnorm:
                layers.append(BatchNorm1d)
            layers.append(ReLU())
    if multiobjective:
        shared_module=Sequential(*layers)
        task1_module = Sequential(Linear(dims[-1], 1),Sigmoid())
        task2_module = Sequential(Linear(dims[-1], 1), Sigmoid())
        task3_module= Sequential(Linear(dims[-1], 1), Sigmoid())
        task4_module= Sequential(Linear(dims[-1], 1), Sigmoid())
        params = [
            *shared_module.parameters(),
            *task1_module.parameters(),
            *task2_module.parameters(),
            *task3_module.parameters(),
            *task4_module.parameters(),
        ]
    else:
        layers.append(Linear(dims[-1],1))
        layers.append(Sigmoid())
        model = Sequential(*layers)
        params=model.parameters()
    if positive_weight == 1:
        criterion = nn.BCELoss()
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([positive_weight], device=torch.device("cpu"))
        )


    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=l2_pen)
    model_save_path = result_folder / Path("checkpoint.pt")
    if multiobjective:
        train_loss_list, val_loss_list, auc_list, ap_list = train_multiobjective(
            shared_module=shared_module,
            task1_module=task1_module,
            task2_module=task2_module,
            task3_module=task3_module,
            task4_module=task4_module,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            n_epochs=n_epochs,
            criterion=criterion,
            model_save_path=model_save_path,
            lr_strategy=lr_strategy,
            mask_prob=mask_prob,
            big_embedding=big_embedding,
        )
    else:
        train_loss_list, val_loss_list, auc_list, ap_list = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            n_epochs=n_epochs,
            criterion=criterion,
            model_save_path=model_save_path,
            lr_strategy=lr_strategy,
            mask_prob=mask_prob,
            big_embedding=big_embedding,
        )
    save_plot_path = result_folder / Path("summary_plot.png")
    save_plot(
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list,
        auc_list=auc_list,
        ap_list=ap_list,
        save_plot_path=save_plot_path,
    )
    best_epoch = np.argmax(ap_list)
    args_dict["best_epoch"] = str(best_epoch + 1)
    args_dict["best_ap"] = str(ap_list[best_epoch])
    with open(result_folder / Path("summary_dict.json"), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

if __name__ == "__main__":
    app()
