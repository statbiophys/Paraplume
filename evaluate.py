import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import typer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential, Sigmoid
from torch_dataset import ParatopeDataset, ParatopeMultiObjectiveDataset
from tqdm import tqdm

app = typer.Typer(add_completion=False)

def create_dataloader(dataset_dict:Dict,residue_embeddings:torch.Tensor, batch_size=10, shuffle:bool=False, alpha:str="4.5")->torch.utils.data.dataloader.DataLoader:
    """Take dataset_dict and embeddings and return dataloader.

    Args:
        dataset_dict (Dict): _description_
        residue_embeddings (torch.Tensor): _description_
        batch_size (int, optional): _description_. Defaults to 10.

    Returns:
        torch.utils.data.dataloader.DataLoader: Dataloader to use for training.
    """
    dataset = ParatopeDataset(dataset_dict=dataset_dict, residue_embeddings=residue_embeddings, alpha=alpha)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader

def test(
    model,
    test_loader,
    big_embedding=False,
):
    # Training loop
    device = torch.device("cpu")
    all_outputs = []
    all_targets = []
    outputs_and_labels = {}
    with torch.no_grad():
        for i, (
            embedding,
            labels,
            len_heavy,
            len_light,
        ) in tqdm(enumerate(test_loader)):
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
            output=model(embedding)
            output=output.view(-1)

            # Convert the tensors to cpu and then to numpy arrays for AUC and ROC calculation
            output = output.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            all_outputs.extend(output)
            all_targets.extend(labels)

    # Converting lists to numpy arrays for AUC and ROC calculation
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets).astype(int)
    # Calculate the AUC score
    auc = roc_auc_score(all_targets, all_outputs)
    ap = average_precision_score(all_targets, all_outputs)

    return outputs_and_labels, auc, ap


@app.command()
def main(
    result_folder: Path = typer.Argument(
        ...,
        help="Path of model.",
        show_default=False,
    ),
    test_folder_path: Path = typer.Argument(
        ...,
        help="Path of testloader.",
        show_default=False,
    ),
    big_embedding:bool=typer.Option(
        False,"--bigembedding", help=("Whether to use big embeddings or not.")
    ),
    multiobjective:bool=typer.Option(
        False,"--multiobjective", help=("Whether to use multiobjective or not.")
    ),
) -> None:
    model_path = result_folder / Path("checkpoint.pt")
    print(result_folder.as_posix())
    args_dict = {
        "model_path": str(model_path),
        "test_folder_path": str(test_folder_path),
        "result_folder": str(result_folder),
    }
    with open(result_folder/Path("summary_dict.json")) as f:
        summary_dict=json.load(f)
    dims=summary_dict["dims"]
    dropouts=summary_dict["dropouts"]
    batchnorm = summary_dict["batchnorm"]
    batch_size = summary_dict["batch_size"]
    alpha=summary_dict["alpha"]
    with open(test_folder_path / Path("dict.json")) as f :
        dict_test = json.load(f)
    test_embeddings = torch.load(test_folder_path / Path("embeddings.pt"), weights_only=True)
    test_loader = create_dataloader(dataset_dict=dict_test, residue_embeddings=test_embeddings, batch_size=batch_size, alpha=alpha)
    #torch.save(test_loader, result_folder / Path(f'test_dataloader_batchsize_{batch_size}.pkl'))
    dims=dims.split(",")
    dims=[int(each) for each in dims]
    dropouts=dropouts.split(",")
    dropouts=[float(each) for each in dropouts]
    print(dims)
    if big_embedding :
        input_size = 2528
    else:
        input_size = 2048
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
        model=Sequential(Sequential(*layers), Sequential(Linear(dims[-1],1),Sigmoid()))
    else:
        layers.append(Linear(dims[-1],1))
        layers.append(Sigmoid())
        model = Sequential(*layers)

    print("LOADING MODEL")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("RETRIEVING RESULTS")
    outputs_and_labels, auc, ap = test(
        model=model,
        test_loader=test_loader,
        big_embedding=big_embedding,
    )
    args_dict["ap"]=ap
    args_dict["auc"]=auc
    args_dict["detailed_results"]=outputs_and_labels

    print("SAVING RESULTS")
    with open(result_folder / Path("results_dict.json"), "w") as json_file:
        json.dump(args_dict, json_file, indent=4)


if __name__ == "__main__":
    app()
