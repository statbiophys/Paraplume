from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils import get_other_labels, read_pdb_to_dataframe


class ParatopeDataset(Dataset):

    def __init__(
        self, dataset_dict: Dict, embeddings: torch.Tensor, alpha: str = "4.5",alphas:Optional[List]=None,mode:str="train",

    ):
        self.mode=mode
        self.dataset_dict = dataset_dict
        self.embeddings = embeddings
        self.alpha = alpha
        self.alphas=alphas

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, index):
        main_labels_heavy = self.dataset_dict[str(index)][f"H_id labels {self.alpha}"]
        main_labels_light = self.dataset_dict[str(index)][f"L_id labels {self.alpha}"]

        numbers_heavy = self.dataset_dict[str(index)]["H_id numbers"]
        numbers_light = self.dataset_dict[str(index)]["L_id numbers"]

        pdb_code = self.dataset_dict[str(index)]["pdb_code"]

        embedding = self.embeddings[index, :, :]

        main_labels_paired = main_labels_heavy + main_labels_light
        main_labels = torch.FloatTensor(
            F.pad(
                torch.FloatTensor(main_labels_paired),
                (0, 280 - len(torch.FloatTensor(main_labels_paired))),
                "constant",
                0,
            )
        )
        len_heavy = len(main_labels_heavy)
        len_light = len(main_labels_light)

        convex_hull_heavy = self.dataset_dict[str(index)]["H_id convex_hull"]
        convex_hull_light = self.dataset_dict[str(index)]["L_id convex_hull"]
        convex_hull_paired = convex_hull_heavy + convex_hull_light
        convex_hull_paired = 1 - np.array(convex_hull_paired) / np.max(
            convex_hull_paired
        )
        convex_hull_padded = torch.FloatTensor(
            F.pad(
                torch.FloatTensor(convex_hull_paired),
                (0, 280 - len(torch.FloatTensor(convex_hull_paired))),
                "constant",
                0,
            )
        )

        distances_heavy = self.dataset_dict[str(index)]["H_id distances"]
        distances_light = self.dataset_dict[str(index)]["L_id distances"]
        distances_paired = distances_heavy + distances_light
        distances_paired = np.array(distances_paired)
        distances_padded = torch.FloatTensor(
            F.pad(
                torch.FloatTensor(distances_paired),
                (0, 280 - len(torch.FloatTensor(distances_paired))),
                "constant",
                0,
            )
        )
        labels_list = get_other_labels(self.dataset_dict, index, alphas=self.alphas)


        if self.mode=="train":
            return (
                embedding,
                main_labels,
                len_heavy,
                len_light,
                convex_hull_padded,
                distances_padded,
                *labels_list,
            )
        if self.mode=="test":
            return (embedding,
                main_labels,
                len_heavy,
                len_light)
        return (embedding,
            main_labels,
            len_heavy,
            len_light,
            pdb_code,
            numbers_heavy,
            numbers_light,
            convex_hull_padded,
            distances_padded)




class AminoAcidGraphEGNN(Dataset):
    def __init__(
        self,
        dataset_dict: Dict,
        embeddings: torch.Tensor,
        csv: pd.DataFrame,
        alpha: str = "4.5",
        pdb_folder_path="/home/gathenes/paragraph_benchmark/abb3_pdbs_renumbered",
        features="one-hot",
    ):
        self.dataset_dict = dataset_dict
        self.embeddings = embeddings
        self.alpha = alpha
        self.csv = csv
        self.pdb_folder_path = pdb_folder_path
        self.features = features
        self.noise=0

    def __len__(self):
        return self.embeddings.shape[0]

    def __add_noise__(self, noise=0):
        self.noise=noise



    def __getitem__(self, index):
        # Load pdb code and chains
        pdb = self.dataset_dict[str(index)]["pdb_code"]
        pdb_path = f"{self.pdb_folder_path}/{pdb}.pdb"
        heavy_chain = self.csv.query("pdb == @pdb")["Hchain"].values[0]
        light_chain = self.csv.query("pdb == @pdb")["Lchain"].values[0]
        chains = [heavy_chain, light_chain]
        # Filter the PB data
        df_pdb = read_pdb_to_dataframe(pdb_path).query("atom_name == 'CA' and chain_id.isin(@chains)")
        df_pdb["residue_letter"]=df_pdb["residue_name"].map(triplet_to_one_letter)




        # Separate heavy and light chains
        heavy_df = df_pdb.query("chain_id == @heavy_chain")
        light_df = df_pdb.query("chain_id == @light_chain")
        heavy_res_dict = {
            res_num: idx for idx, res_num in enumerate(heavy_df["IMGT"])
        }
        light_res_dict = {
            res_num: idx for idx, res_num in enumerate(light_df["IMGT"])
        }

        cdrs = (
            list(range(25, 40 + 1))
            + list(range(54, 67 + 1))
            + list(range(103, 119 + 1))
        )
        heavy_cdr = heavy_df.query("residue_number in @cdrs")
        light_cdr = light_df.query("residue_number in @cdrs")

        # add noise in pdb
        heavy_noise_cdr1 = self.noise*0.87
        heavy_noise_cdr2 = self.noise*0.75
        heavy_noise_cdr3 = self.noise*2.42

        light_noise_cdr1 = self.noise*0.61
        light_noise_cdr2 = self.noise*0.39
        light_noise_cdr3 = self.noise*0.93

        heavy_cdr = add_noise(heavy_cdr,noise_cdr1=heavy_noise_cdr1, noise_cdr2=heavy_noise_cdr2,noise_cdr3=heavy_noise_cdr3)
        light_cdr = add_noise(light_cdr,noise_cdr1=light_noise_cdr1, noise_cdr2=light_noise_cdr2,noise_cdr3=light_noise_cdr3)

        # Prepare labels and embeddings
        labels_heavy = torch.tensor(
            self.dataset_dict[str(index)][f"H_id labels {self.alpha}"],
            dtype=torch.float32,
        )
        labels_light = torch.tensor(
            self.dataset_dict[str(index)][f"L_id labels {self.alpha}"],
            dtype=torch.float32,
        )
        embedding = self.embeddings[index]

        # Collect features (feats) and coordinates (coors)
        node_features = []
        node_coords = []
        node_labels = []
        imgt_numbers_heavy=[]
        # Heavy chain features and coordinates
        for i, (res, name) in enumerate(
            zip(heavy_cdr["IMGT"].tolist(), heavy_cdr["residue_letter"].tolist())
        ):
            imgt_numbers_heavy.append(res)
            res_index = heavy_res_dict[res]

            if self.features == "ablang":
                node_features.append(
                    embedding[1 + res_index][2048:]
                )  # Assuming embeddings are (num_nodes, 2048)
            elif self.features == "T5":
                node_features.append(embedding[1 + res_index][1024:2048])
            elif self.features == "all-llm":
                node_features.append(embedding[1 + res_index])
            else :
                node_feature = torch.tensor([int(each == name) for each in amino_acids]+[1, 0])
                node_features.append(node_feature)

            node_coords.append(heavy_cdr.iloc[i][["x_coord", "y_coord", "z_coord"]].astype(float).values)
            node_labels.append(labels_heavy[res_index])

        # Light chain features and coordinates
        imgt_numbers_light=[]
        for i, (res, name) in enumerate(
            zip(light_cdr["IMGT"].tolist(), light_cdr["residue_letter"].tolist())
        ):
            imgt_numbers_light.append(res)
            res_index = light_res_dict[res]

            if self.features == "ablang":
                node_features.append(
                    embedding[len(labels_heavy) + 4 + res_index][2048:]
                )
            elif self.features == "T5":
                node_features.append(
                    embedding[len(labels_heavy) + 2 + res_index][1024:2048]
                )
            elif self.features == "all-llm":
                node_features.append(
                    torch.cat(
                        [
                            embedding[len(labels_heavy) + 2 + res_index][:2048],
                            embedding[len(labels_heavy) + 4 + res_index][2048:],
                        ],
                        dim=1,
                    )
                )
            else:
                node_feature = torch.tensor([int(each == name) for each in amino_acids]+[0, 1])
                node_features.append(node_feature)

            node_coords.append(light_cdr.iloc[i][["x_coord", "y_coord", "z_coord"]].astype(float).values)
            node_labels.append(labels_light[res_index])

        # Convert features and coordinates to tensors and reshape as required
        feats = torch.stack(node_features)  # Shape: (1, num_samples, num_feats)
        labels = torch.stack(node_labels)
        node_coords = np.array(node_coords)
        coors = torch.tensor(
            node_coords, dtype=torch.float32
        )  # Shape: (1, num_samples, 3)

        # Calculate pairwise distances for edges
        antibody_coords = np.array(node_coords)
        distances = np.linalg.norm(
            antibody_coords[:, np.newaxis] - antibody_coords, axis=2
        )

        # Generate the edges tensor as specified
        edges = torch.tensor(distances, dtype=torch.float32).unsqueeze(
            -1
        )  # Shape: (1, num_samples, num_samples, 1)
        edges = (edges < 10).int()
        num_samples = edges.shape[0]
        edges[torch.arange(num_samples), torch.arange(num_samples), 0] = 0

        return (feats.to(torch.float), coors.to(torch.float), edges.to(torch.float)), labels.to(torch.float),pdb,imgt_numbers_heavy, imgt_numbers_light


def create_dataloader(dataset_dict:Dict,embeddings:torch.Tensor, batch_size=10, shuffle:bool=False, alpha:str="4.5", alphas:Optional[List]=None, mode="train")->torch.utils.data.dataloader.DataLoader:
    """Take dataset_dict and embeddings and return dataloader.

    Args:
        dataset_dict (Dict): _description_
        embeddings (torch.Tensor): _description_
        batch_size (int, optional): _description_. Defaults to 10.

    Returns:
        torch.utils.data.dataloader.DataLoader: Dataloader to use for training.
    """
    if mode=="train":
        shuffle=True
    else:
        shuffle = False
    dataset=ParatopeDataset(dataset_dict=dataset_dict, embeddings=embeddings, alpha=alpha, alphas=alphas, mode=mode)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader

def create_graph_dataloader(
    pdb_folder_path: Path,
    dataset_dict: Dict,
    embeddings: torch.Tensor,
    csv,
    batch_size=1,
    shuffle: bool = False,
    features="one-hot",
) -> torch.utils.data.dataloader.DataLoader:
    """Take dataset_dict and embeddings and return dataloader.

    Args:
        dataset_dict (Dict): _description_
        residue_embeddings (torch.Tensor): _description_
        batch_size (int, optional): _description_. Defaults to 10.

    Returns:
        torch.utils.data.dataloader.DataLoader: Dataloader to use for training.
    """
    dataset = AminoAcidGraphEGNN(
        dataset_dict=dataset_dict,
        embeddings=embeddings,
        csv=csv,
        pdb_folder_path=pdb_folder_path,
        features=features,
    )
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader

def add_noise(df,noise_cdr1=0, noise_cdr2=0, noise_cdr3=0):

# Define CDR ranges
    cdr1 = list(range(25, 40 + 1))
    cdr2 = list(range(54, 67 + 1))
    cdr3 = list(range(103, 119 + 1))

    df_cdr1=df.query("residue_number.isin(@cdr1)")
    df_cdr2=df.query("residue_number.isin(@cdr2)")
    df_cdr3=df.query("residue_number.isin(@cdr3)")

    # Apply different noise levels to each CDR in the heavy chain
    df_cdr1['x_coord'] += np.random.normal(0, noise_cdr1, size=df_cdr1['x_coord'].shape)
    df_cdr1['y_coord'] += np.random.normal(0, noise_cdr1, size=df_cdr1['y_coord'].shape)
    df_cdr1['z_coord'] += np.random.normal(0, noise_cdr1, size=df_cdr1['z_coord'].shape)

    df_cdr2['x_coord'] += np.random.normal(0, noise_cdr2, size=df_cdr2['x_coord'].shape)
    df_cdr2['y_coord'] += np.random.normal(0, noise_cdr2, size=df_cdr2['y_coord'].shape)
    df_cdr2['z_coord'] += np.random.normal(0, noise_cdr2, size=df_cdr2['z_coord'].shape)

    df_cdr3['x_coord'] += np.random.normal(0, noise_cdr3, size=df_cdr3['x_coord'].shape)
    df_cdr3['y_coord'] += np.random.normal(0, noise_cdr3, size=df_cdr3['y_coord'].shape)
    df_cdr3['z_coord'] += np.random.normal(0, noise_cdr3, size=df_cdr3['z_coord'].shape)

    return df


triplet_to_one_letter = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}

amino_acids = [
    'A', 'R', 'N', 'D', 'C',
    'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
