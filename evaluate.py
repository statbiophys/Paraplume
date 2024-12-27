import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import typer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential, Sigmoid
from torch_dataset import create_dataloader
from tqdm import tqdm
from utils import get_embedding

app = typer.Typer(add_completion=False)

def test(
    model,
    test_loader,
    embedding_models_list=["igT5"],
    threshold=0,
    best_val_mcc_threshold=0,
    best_val_f1_threshold=0,
):
    # Training loop
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    all_outputs = []
    all_targets = []
    outputs_and_labels = {}
    with torch.no_grad():
        for _, (
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
                emb = get_embedding(embedding=embedding[i], embedding_models=embedding_models_list, heavy=heavy, light=light)
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
    all_predictions_f1 = [1 if x >= threshold else 0 for x in all_outputs]
    f1 = f1_score(all_targets, all_predictions_f1)
    all_predictions_mcc=[1 if x >= threshold else 0 for x in all_outputs]
    mcc = matthews_corrcoef(all_targets, all_predictions_mcc)

    all_predictions_f1 = [1 if x >= 0.5 else 0 for x in all_outputs]
    f1_05 = f1_score(all_targets, all_predictions_f1)
    all_predictions_mcc=[1 if x >= 0.5 else 0 for x in all_outputs]
    mcc_05 = matthews_corrcoef(all_targets, all_predictions_mcc)

    all_predictions_mcc=[1 if x >= best_val_mcc_threshold else 0 for x in all_outputs]
    best_val_mcc = matthews_corrcoef(all_targets, all_predictions_mcc)

    all_predictions_f1=[1 if x >= best_val_f1_threshold else 0 for x in all_outputs]
    best_val_f1 = matthews_corrcoef(all_targets, all_predictions_mcc)

    return outputs_and_labels, auc, ap, f1, mcc, f1_05, mcc_05, best_val_mcc, best_val_f1


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
) -> None:
    model_path = result_folder / Path("checkpoint.pt")
    print(result_folder.as_posix())
    args_dict = {
        "model_path": str(model_path),
        "test_folder_path": str(test_folder_path),
        "result_folder": str(result_folder),
    }
    with open(result_folder/Path("summary_dict.json"), encoding="utf-8") as f:
        summary_dict=json.load(f)
    dims=summary_dict["dims"]
    dropouts=summary_dict["dropouts"]
    batchnorm = summary_dict["batchnorm"]
    batch_size = summary_dict["batch_size"]
    alpha=summary_dict["alpha"]
    with open(test_folder_path / Path("dict.json"), encoding="utf-8") as f :
        dict_test = json.load(f)
    test_embeddings = torch.load(test_folder_path / Path("embeddings.pt"), weights_only=True)
    test_loader = create_dataloader(dataset_dict=dict_test, embeddings=test_embeddings, batch_size=batch_size, alpha=alpha, mode="test")
    dims=dims.split(",")
    dims=[int(each) for each in dims]
    dropouts=dropouts.split(",")
    dropouts=[float(each) for each in dropouts]
    print(dims)
    input_size = summary_dict["input_size"]
    embedding_models=summary_dict["embedding_models"]
    if embedding_models=='all':
        embedding_models="ablang2,igbert,igT5,esm,antiberty,prot-t5"
    embedding_models_list = embedding_models.split(",")
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
    model=Sequential(Sequential(*layers), Sequential(Linear(dims[-1],1),Sigmoid()))


    print("LOADING MODEL")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("RETRIEVING RESULTS")
    threshold = float(summary_dict["threshold_youden"])
    best_val_mcc_threshold = float(summary_dict["best_val_mcc_threshold"])
    best_val_f1_threshold = float(summary_dict["best_val_f1_threshold"])
    outputs_and_labels, auc, ap, f1, mcc, f1_05, mcc_05, best_val_mcc, best_val_f1 = test(
        model=model,
        test_loader=test_loader,
        embedding_models_list=embedding_models_list,
        threshold=threshold,
        best_val_mcc_threshold=best_val_mcc_threshold,
        best_val_f1_threshold=best_val_f1_threshold,
    )
    args_dict["ap"]=ap
    args_dict["auc"]=auc
    args_dict["f1"]=f1
    args_dict["mcc"]=mcc
    args_dict["f1_05"]=f1_05
    args_dict["mcc_05"]=mcc_05
    args_dict["mcc_best_val"]=best_val_mcc
    args_dict["f1_best_val"]=best_val_f1
    args_dict["detailed_results"]=outputs_and_labels


    print("SAVING RESULTS")
    with open(result_folder / Path(f"{test_folder_path.stem}_results_dict.json"), "w", encoding="utf-8") as json_file:
        json.dump(args_dict, json_file, indent=4)


if __name__ == "__main__":
    app()
