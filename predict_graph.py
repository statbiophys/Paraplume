import json
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import typer
from biopandas.pdb import PandasPdb
from models import MLP
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential, Sigmoid
from torch_dataset import ParatopePredictDataset
from tqdm import tqdm
from utils import format_pdb

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
    model_name="pred",
):
    # Training loop
    device = torch.device("cpu")
    pdb_dict = {}

    with torch.no_grad():
        total_dataframe = pd.DataFrame()
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
                pdb_dict[pdb_code]["heavy prediction"][inverse_heavy[i][0]]=float(output[i])
                pdb_dict[pdb_code]["heavy label"][inverse_heavy[i][0]]=float(labels[i])
            for i in range(light):
                pdb_dict[pdb_code]["light prediction"][inverse_light[i][0]]=float(output[heavy+i])
                pdb_dict[pdb_code]["light label"][inverse_light[i][0]]=float(labels[heavy+i])
            pdb_path = f"/home/gathenes/all_structures/imgt/{pdb_code}.pdb"
            data_pdb = format_pdb(f"/home/gathenes/all_structures/imgt/{pdb_code}.pdb").rename(columns={"Chain":"chain_id", "Res_Num":"residue_number"})
            light_chain, heavy_chain, antigen_chain = chains.query("pdb==@pdb_code")[["Lchain", "Hchain", "antigen_chain"]].values[0]

            data_pdb["IMGT"] = data_pdb["residue_number"].str.replace(r'[a-zA-Z]$', '', regex=True).astype(int)


            data_pdb_heavy=data_pdb.query("chain_id==@heavy_chain")
            data_pdb_light=data_pdb.query("chain_id==@light_chain")

            data_pdb_light=data_pdb_light.query("IMGT<128")
            data_pdb_heavy=data_pdb_heavy.query("IMGT<129")

            data_pdb_antigen=data_pdb.query("chain_id==@antigen_chain")


            data_pdb_heavy["b_factor"]=data_pdb_heavy["residue_number"].map(pdb_dict[pdb_code]["heavy prediction"])
            data_pdb_light["b_factor"]=data_pdb_light["residue_number"].map(pdb_dict[pdb_code]["light prediction"])
            data_pdb_heavy["occupancy"]=data_pdb_heavy["residue_number"].map(pdb_dict[pdb_code]["heavy label"])
            data_pdb_light["occupancy"]=data_pdb_light["residue_number"].map(pdb_dict[pdb_code]["light label"])
            data_pdb_antigen["b_factor"]=0
            data_pdb_antigen["occupancy"]=0
            new_data_pdb = pd.concat([data_pdb_heavy, data_pdb_light, data_pdb_antigen])
            new_data_pdb["record_name"]="ATOM"
            new_data_pdb["blank_1"]=""
            new_data_pdb["blank_2"]=""
            new_data_pdb["blank_3"]=""
            new_data_pdb["blank_4"]=""
            new_data_pdb["alt_loc"]=""
            new_data_pdb["insertion"]=""
            new_data_pdb["model"]=1
            new_data_pdb["segment_id"]=""
            new_data_pdb["charge"]=0
            new_data_pdb["line_idx"]=""
            new_data_pdb = new_data_pdb.rename(columns={"Atom_Num":"atom_number","AA":"residue_name","Atom_Name":"atom_name","x":"x_coord","y":"y_coord","z":"z_coord", "Atom_type":"element_symbol"})
            new_data_pdb = new_data_pdb.astype({"x_coord":float,"y_coord":float,"z_coord":float})

            atomic_df = PandasPdb().read_pdb(pdb_path)
            atomic_df = atomic_df.get_model(1)
            if len(atomic_df.df["ATOM"]) == 0:
                raise ValueError(f"No model found for index: {1}")
            atomic_df.df["ATOM"] = new_data_pdb
            atomic_df.to_pdb(save_path / f"{pdb_code}.pdb", records = ["ATOM"])

            chain_ids=[heavy_chain, light_chain]
            antibody_only = atomic_df.df["ATOM"].query("chain_id.isin([@chain_ids])")
            antibody_only.rename(columns={"occupancy":"labels","b_factor":model_name})
            total_dataframe=pd.concat([total_dataframe, antibody_only])
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
) -> None:
    chains=pd.read_csv(chains_path)
    result_folder=model_path.parents[0]
    print(result_folder.as_posix())
    with open(test_folder_path / Path("dict.json")) as f :
        dict_test = json.load(f)
    test_embeddings = torch.load(test_folder_path / Path("embeddings.pt"), weights_only=True)
    test_loader = create_dataloader(dataset_dict=dict_test, residue_embeddings=test_embeddings, batch_size=1, alpha=4.5)
    with open(result_folder / Path("summary_dict.json")) as f:
        summary_dict = json.load(f)
    dims=summary_dict["dims"].split(",")
    dims=[int(each) for each in dims]
    dropouts=summary_dict["dropouts"]
    dropouts=dropouts.split(",")
    dropouts=[float(each) for each in dropouts]
    batchnorm=summary_dict["batchnorm"]
    big_embedding=summary_dict["big_embedding"]
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
    model=Sequential(Sequential(*layers), Sequential(Linear(dims[-1],1),Sigmoid()))
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
        save_path=save_path,
        model_name=str(result_folder.name)
    )


if __name__ == "__main__":
    app()
