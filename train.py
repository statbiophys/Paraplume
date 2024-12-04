import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from models import EarlyStopping
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential, Sigmoid
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch_dataset import create_dataloader
from torchjd import mtl_backward
from torchjd.aggregation import UPGrad
from tqdm import tqdm
from utils import save_plot

app = typer.Typer(add_completion=False)


def train_multiobjective(
    shared_module,
    main_task_module,
    convex_task_module,
    distance_task_module,
    other_tasks,
    train_loader,
    val_loader,
    optimizer,
    n_epochs=3,
    model_save_path=Path("/home/gathenes/all_structures/data_high_qual/checkpoint.pt"),
    criterion=nn.BCELoss(),
    mask_prob:float=0,
    lr_strategy:Optional[str]=None,  # Can be "step", "plateau", or "cosine"
    big_embedding=False,
    convex=False,
    distance=False,
    patience=0,
):
    A=UPGrad()
    device = torch.device("cpu")
    train_loss_list = []
    val_loss_list = []
    auc_list = []
    ap_list = []

    criterion_cel = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=patience, path=model_save_path, best_score=0)
    if lr_strategy == "step":
        scheduler = StepLR(optimizer, step_size=40)
    elif lr_strategy == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=4)
    elif lr_strategy == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=25)
    elif lr_strategy:
        raise ValueError(f"Unknown scheduler type: {lr_strategy}")
    else :
        scheduler=None
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        shared_module.train()
        for i, (embedding, main_labels,len_heavy, len_light,  labels_convex, labels_distance, *other_labels) in enumerate(tqdm(train_loader)):
            embedding, main_labels, labels_convex, labels_distance = embedding.to(device), main_labels.to(device),labels_convex.to(device),  labels_distance.to(device)

            embedding_list=[]

            convex_labels_list = []
            main_labels_list = []
            distance_labels_list = []

            other_labels_list = []
            for i in range(len(other_labels)):
                other_labels_list.append([])


            for i in range(len_heavy.shape[-1]):
                heavy, light = len_heavy[i], len_light[i]
                ran = list(range(1,heavy+1))+list(range(heavy+2, heavy+light+2))
                emb=embedding[i][ran,:2048]
                if big_embedding:
                    ran2 = list(range(1,heavy+1))+list(range(heavy+4, heavy+light+4))
                    emb2=embedding[i][ran2,2048:]
                    emb=torch.cat([emb,emb2], dim=1)

                lab_convex=labels_convex[i][:heavy+light]
                lab_main=main_labels[i][:heavy+light]
                lab_distance=labels_distance[i][:heavy+light]

                convex_labels_list.append(lab_convex)
                main_labels_list.append(lab_main)
                distance_labels_list.append(lab_distance)

                for l,each in enumerate(other_labels):
                    other_label = each[i][:heavy+light]
                    other_labels_list[l].append(other_label)

                if random.random()<mask_prob:
                    drop_mask = torch.rand(emb.size(), device=emb.device) >= mask_prob
                    emb=emb * drop_mask.float()
                embedding_list.append(emb)

            embedding = torch.cat(embedding_list, dim=0)
            labels_convex = torch.cat(convex_labels_list, dim=0)
            main_labels = torch.cat(main_labels_list, dim=0)
            labels_distance = torch.cat(distance_labels_list, dim=0)

            other_labels=[torch.cat(each, dim=0) for each in other_labels_list]


            optimizer.zero_grad()
            features = shared_module(embedding)

            output_main = main_task_module(features).view(-1)
            output_convex = convex_task_module(features).view(-1)
            output_distance = distance_task_module(features).view(-1)

            loss_main = criterion(output_main, main_labels)
            loss_convex = criterion_cel(output_convex, labels_convex)
            loss_distances = criterion_cel(output_distance, labels_distance)

            other_outputs=[]
            other_tasks_params=[]
            for other_task in other_tasks:
                other_outputs.append(other_task(features).view(-1))
                other_tasks_params+=[other_task.parameters()]

            other_losses=[]
            for other_output, other_label in zip(other_outputs, other_labels):
                other_losses.append(criterion(other_output, other_label))

            losses=[loss_main]
            task_params = [main_task_module.parameters()]

            if convex:
                losses.append(loss_convex)
                task_params+=[convex_task_module.parameters()]
            if distance:
                losses.append(loss_distances)
                task_params+=[distance_task_module.parameters()]
            losses+=[*other_losses]
            task_params+=[*other_tasks_params]
            mtl_backward(
                losses=losses,
                features=features,
                tasks_params=task_params,
                shared_params=shared_module.parameters(),
                A=A,
            )
            optimizer.step()
            train_loss += loss_main.item() * embedding.size(0)
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)
        val_loss = 0
        all_outputs = []
        all_targets = []
        print(f"Saving train loss and epoch, epoch={epoch}, train_loss={train_loss}")
        with torch.no_grad():
            model = nn.Sequential(
            shared_module,
            main_task_module
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
                output_sigmoid = torch.sigmoid(output).view(-1).detach().cpu().numpy()
                output=output.view(-1)
                loss = criterion(output, labels)
                val_loss += loss.item() * embedding.size(0)
                output = output.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                all_outputs.extend(output_sigmoid)
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
        if patience>0:
            early_stopping(-ap, model)
            if early_stopping.early_stop:
                print(f"Early stopping, last epoch = {epoch}")
                break
        else :
            torch.save(model.state_dict(), model_save_path.as_posix())
        if lr_strategy in ["step","cosine"]:
                scheduler.step()  # StepLR scheduler: decrease lr after `step_size` epochs
        if lr_strategy == "plateau":
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
        4.5, "--alpha", help="Alpha distance to use for main objective. Default to 4.5."
    ),
    seed:int=typer.Option(
        0, "--seed", help="Seed to use for training."
    ),
    l2_pen:float=typer.Option(
        0, "--l2-pen", help="L2 penalty to use for the model weights."
    ),
    big_embedding:bool=typer.Option(
        False,"--bigembedding", help="Whether to use big embeddings or not."
    ),
    convex:bool=typer.Option(
        False,"--convex", help="Whether to use convex hull labels."
    ),
    distance:bool=typer.Option(
        False,"--distance", help="Whether to use distance labels or not."
    ),
    alphas:str=typer.Option(
        "-", "--alphas", help="Whether to use different alphas labels to help main label. \
        Defaults to empty."
    ),
    patience:int=typer.Option(
        0, "--patience", help="Patience to use for early stopping. 0 means no early stopping. \
        Defaults to 0."
    )
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
        "big_embedding":big_embedding,
        "convex":convex,
        "distance":distance,
        "alphas":alphas,
        "patience":patience,
    }
    if alphas=="-":
        alphas=None
    else :
        alphas=alphas.split(",")
    if seed>0:
        torch.manual_seed(seed)
    with open(train_folder_path / Path("dict.json"), encoding="utf-8") as f :
        dict_train = json.load(f)
    train_embeddings = torch.load(train_folder_path / Path("embeddings.pt"), weights_only=True)

    with open(val_folder_path / Path("dict.json"), encoding="utf-8") as f :
        dict_val = json.load(f)
    val_embeddings = torch.load(val_folder_path / Path("embeddings.pt"), weights_only=True)

    train_loader = create_dataloader(dataset_dict=dict_train, embeddings=train_embeddings, batch_size=batch_size, alpha=alpha, alphas=alphas, mode="train")
    val_loader = create_dataloader(dataset_dict=dict_val, embeddings=val_embeddings, batch_size=batch_size, alpha=alpha,mode="test")
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
    shared_module=Sequential(*layers)
    main_task_module = Sequential(Linear(dims[-1], 1))
    convex_task_module = Sequential(Linear(dims[-1], 1), Sigmoid())
    distance_task_module= Sequential(Linear(dims[-1], 1), Sigmoid())
    params = [
        *shared_module.parameters(),
        *main_task_module.parameters()]
    if convex:
        params+=[*convex_task_module.parameters()]
    if distance:
        params+=[*distance_task_module.parameters()]

    other_tasks=[]
    if alphas:
        for i in range(len(alphas)):
            other_task=Sequential(Linear(dims[-1], 1))
            params+=[*other_task.parameters()]
            other_tasks.append(other_task)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], device=torch.device("cpu"))
    )
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=l2_pen)
    model_save_path = result_folder / Path("checkpoint.pt")
    train_loss_list, val_loss_list, auc_list, ap_list = train_multiobjective(
        shared_module=shared_module,
        main_task_module=main_task_module,
        convex_task_module=convex_task_module,
        distance_task_module=distance_task_module,
        other_tasks=other_tasks,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=n_epochs,
        criterion=criterion,
        model_save_path=model_save_path,
        lr_strategy=lr_strategy,
        mask_prob=mask_prob,
        big_embedding=big_embedding,
        convex=convex,
        distance=distance,
        patience=patience,
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
    with open(result_folder / Path("summary_dict.json"), 'w', encoding="utf-8") as json_file:
        json.dump(args_dict, json_file, indent=4)

if __name__ == "__main__":
    app()
