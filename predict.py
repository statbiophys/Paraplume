import json
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import typer
from biopandas.pdb import PandasPdb
from create_dataset import create_dataloader
from models import MLP
from torch_dataset import ParatopePredictDataset
from tqdm import tqdm
from utils import read_pdb_to_dataframe

warnings.filterwarnings("ignore")
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
    dataset = ParatopePredictDataset(dataset_dict=dataset_dict, residue_embeddings=residue_embeddings, alpha=alpha)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader

def test(
    model,
    test_loader,
    chains:pd.DataFrame,
    save_path: Path,
):
    # Training loop
    device = torch.device("cpu")
    pdb_dict = {}
    with torch.no_grad():
        for i, (
            embedding,
            labels,
            heavy,
            light,
            pdb_code,
            inverse_heavy,
            inverse_light,
        ) in tqdm(enumerate(test_loader)):
            pdb_code=pdb_code[0]
            pdb_dict[pdb_code]={"heavy prediction":{},"heavy label":{}, "light label":{}, "light prediction":{}}
            embedding, labels = embedding.to(device), labels.to(device)
            embedding, labels=embedding[0], labels[0]
            ran = list(range(1,heavy+1))+list(range(heavy+2, heavy+light+2))
            embedding=embedding[ran]
            output=model(embedding)
            output=output.view(-1)
            for i in range(heavy):
                pdb_dict[pdb_code]["heavy prediction"][int(inverse_heavy[i].detach().cpu().numpy())]=float(output[i].detach().cpu().numpy())
                pdb_dict[pdb_code]["heavy label"][int(inverse_heavy[i].detach().cpu().numpy())]=float(labels[i].detach().cpu().numpy())
            for i in range(light):
                pdb_dict[pdb_code]["light prediction"][int(inverse_light[i].detach().cpu().numpy())]=float(output[heavy+i].detach().cpu().numpy())
                pdb_dict[pdb_code]["light label"][int(inverse_light[i].detach().cpu().numpy())]=float(labels[heavy+i].detach().cpu().numpy())
            pdb_path = f"/home/gathenes/all_structures/imgt/{pdb_code}.pdb"
            data_pdb, _ = read_pdb_to_dataframe(pdb_path)
            light_chain, heavy_chain, antigen_chain = chains.query("pdb==@pdb_code")[["Lchain", "Hchain", "antigen_chain"]].values[0]
            data_pdb_heavy=data_pdb.query("chain_id==@heavy_chain")
            data_pdb_light=data_pdb.query("chain_id==@light_chain")
            data_pdb_antigen=data_pdb.query("chain_id==@antigen_chain")
            data_pdb_heavy["b_factor"]=data_pdb_heavy["residue_number"].map(pdb_dict[pdb_code]["heavy prediction"])
            data_pdb_light["b_factor"]=data_pdb_light["residue_number"].map(pdb_dict[pdb_code]["light prediction"])
            data_pdb_heavy["occupancy"]=data_pdb_heavy["residue_number"].map(pdb_dict[pdb_code]["heavy label"])
            data_pdb_light["occupancy"]=data_pdb_light["residue_number"].map(pdb_dict[pdb_code]["light label"])
            data_pdb_antigen["b_factor"]=0
            new_data_pdb = pd.concat([data_pdb_heavy, data_pdb_light, data_pdb_antigen])
            atomic_df = PandasPdb().read_pdb(pdb_path)
            atomic_df = atomic_df.get_model(1)
            if len(atomic_df.df["ATOM"]) == 0:
                raise ValueError(f"No model found for index: {1}")
            atomic_df.df["ATOM"] = new_data_pdb.query("record_name=='ATOM'")
            atomic_df.to_pdb(save_path / f"{pdb_code}.pdb", records = ["ATOM"])
    # Calculate the AUC score


@app.command()
def main(
    model_path: Path = typer.Argument(
        ...,
        help="Path of model.",
        show_default=False,
    ),
    test_folder_path: Path = typer.Argument(
        ...,
        help="Path of testloader.",
        show_default=False,
    ),
    chains_path: Path = typer.Argument(
        ...,
        help="Path of test csv.",
        show_default=False,
    ),
    model_type: str = typer.Option(
        "MLP", "--model-type", "-m", help="Model to use for training."
    ),
    dim1:int=typer.Option(
        1000, "--dim1", help="Dimension of first layer. Defaults to 1000."
    ),
    dim2:int=typer.Option(
        1, "--dim2", help="Dimension of second layer. 1 means no second layer. Defaults to 1."
    ),
    dim3:int=typer.Option(
        1, "--dim3", help="Dimension of second layer. 1 means no second layer. Defaults to 1."
    ),
    batch_norm:bool=typer.Option(
        False, "--batch-norm", help="Whether to use batchnorm or not. Defaults to False."
    ),
    alpha:str=typer.Option(
        4.5, "--alpha", help="Alpha distance to use for labels. Default to 4.5."
    )
) -> None:
    chains=pd.read_csv(chains_path)
    result_folder=model_path.parents[0]
    print(result_folder.as_posix())
    with open(test_folder_path / Path("dict.json")) as f :
        dict_test = json.load(f)
    test_embeddings = torch.load(test_folder_path / Path("embeddings.pt"), weights_only=True)
    test_loader = create_dataloader(dataset_dict=dict_test, residue_embeddings=test_embeddings, batch_size=1, alpha=alpha)
    #torch.save(test_loader, result_folder / Path(f'test_dataloader_batchsize_{batch_size}.pkl'))

    if model_type=="MLP":
        model = MLP(dim1=dim1, dim2=dim2,dim3=dim3, batch_norm=batch_norm)
    else:
        raise ValueError("No model of this name known.")
    print("LOADING MODEL")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print("RETRIEVING RESULTS")
    save_path=result_folder / Path("visualize")
    save_path.mkdir(exist_ok=True, parents=True)
    test(
        model=model,
        test_loader=test_loader,
        chains=chains,
        save_path=save_path
    )


if __name__ == "__main__":
    app()
