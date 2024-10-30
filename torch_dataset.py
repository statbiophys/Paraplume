
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils import format_pdb


def cdr_indices(inverse_number):
    cdr_list = []
    for cdr_border in [(25,40+1),(54,67+1),(103,119+1)]:
        left = cdr_border[0]
        right = cdr_border[1]
        while left not in inverse_number :
            left+=1
        while right not in inverse_number:
            right-=1
        cdr = list(range(inverse_number[left], inverse_number[right]))
        cdr_list.append(cdr)
    return cdr_list

def get_cdr_pm2_indices(inverse_number_heavy, inverse_number_light):
    cdr_heavy_list = cdr_indices(inverse_number_heavy)
    cdr_light_list = cdr_indices(inverse_number_light)

    heavy_indices = cdr_heavy_list[0]+cdr_heavy_list[1]+cdr_heavy_list[2]
    light_indices = cdr_light_list[0]+cdr_light_list[1]+cdr_light_list[2]
    return heavy_indices, light_indices

class ParatopeDataset(Dataset):

    def __init__(self, dataset_dict: Dict, residue_embeddings:torch.Tensor, alpha:str="4.5"):
        self.dataset_dict = dataset_dict
        self.residue_embeddings = residue_embeddings
        self.alpha=alpha

    def __len__(self):
        return self.residue_embeddings.shape[0]

    def __getitem__(self, index):
        labels_heavy_1 = self.dataset_dict[str(index)][f"H_id labels {self.alpha}"]
        labels_light_1 = self.dataset_dict[str(index)][f"L_id labels {self.alpha}"]

        residue_embedding = self.residue_embeddings[index,:,:]

        labels_paired_1 = labels_heavy_1+labels_light_1
        labels_padded_1 = torch.FloatTensor(F.pad(torch.FloatTensor(labels_paired_1), (0, 280-len(torch.FloatTensor(labels_paired_1))),"constant", 0))
        len_heavy = len(labels_heavy_1)
        len_light = len(labels_light_1)

        return residue_embedding, labels_padded_1, len_heavy, len_light

class ParatopeMultiObjectiveDataset(Dataset):

    def __init__(self, dataset_dict: Dict, residue_embeddings:torch.Tensor, alpha:str="4.5"):
        self.dataset_dict = dataset_dict
        self.residue_embeddings = residue_embeddings
        self.alpha=alpha

    def __len__(self):
        return self.residue_embeddings.shape[0]

    def __getitem__(self, index):
        residue_embedding = self.residue_embeddings[index,:,:]

        labels_heavy_0 = self.dataset_dict[str(index)]["H_id labels 4"]
        labels_light_0 = self.dataset_dict[str(index)]["L_id labels 4"]
        len_heavy = len(labels_heavy_0)
        len_light = len(labels_light_0)
        labels_paired_0 = labels_heavy_0+labels_light_0
        labels_padded_0 = torch.FloatTensor(F.pad(torch.FloatTensor(labels_paired_0), (0, 280-len(torch.FloatTensor(labels_paired_0))),"constant", 0))

        labels_heavy_1 = self.dataset_dict[str(index)][f"H_id labels {self.alpha}"]
        labels_light_1 = self.dataset_dict[str(index)][f"L_id labels {self.alpha}"]
        labels_paired_1 = labels_heavy_1+labels_light_1
        labels_padded_1 = torch.FloatTensor(F.pad(torch.FloatTensor(labels_paired_1), (0, 280-len(torch.FloatTensor(labels_paired_1))),"constant", 0))

        labels_heavy_2 = self.dataset_dict[str(index)]["H_id labels 5"]
        labels_light_2 = self.dataset_dict[str(index)]["L_id labels 5"]
        labels_paired_2 = labels_heavy_2+labels_light_2
        labels_padded_2 = torch.FloatTensor(F.pad(torch.FloatTensor(labels_paired_2), (0, 280-len(torch.FloatTensor(labels_paired_2))),"constant", 0))

        labels_heavy_3 = self.dataset_dict[str(index)]["H_id labels 5.5"]
        labels_light_3 = self.dataset_dict[str(index)]["L_id labels 5.5"]
        labels_paired_3 = labels_heavy_3+labels_light_3
        labels_padded_3 = torch.FloatTensor(F.pad(torch.FloatTensor(labels_paired_3), (0, 280-len(torch.FloatTensor(labels_paired_3))),"constant", 0))

        return residue_embedding, len_heavy, len_light, labels_padded_0, labels_padded_1, labels_padded_2, labels_padded_3

class ParatopePredictDataset(Dataset):

    def __init__(self, dataset_dict: Dict, residue_embeddings:torch.Tensor, alpha:str="4.5"):
        self.dataset_dict = dataset_dict
        self.residue_embeddings = residue_embeddings
        self.alpha=alpha

    def __len__(self):
        return self.residue_embeddings.shape[0]

    def __getitem__(self, index):
        labels_heavy_1 = self.dataset_dict[str(index)][f"H_id labels {self.alpha}"]
        labels_light_1 = self.dataset_dict[str(index)][f"L_id labels {self.alpha}"]
        pdb_code = self.dataset_dict[str(index)]["pdb_code"]
        numbers_heavy = self.dataset_dict[str(index)]["H_id numbers"]
        numbers_light = self.dataset_dict[str(index)]["L_id numbers"]

        residue_embedding = self.residue_embeddings[index,:,:]

        labels_paired_1 = labels_heavy_1+labels_light_1
        labels_padded_1 = torch.FloatTensor(F.pad(torch.FloatTensor(labels_paired_1), (0, 280-len(torch.FloatTensor(labels_paired_1))),"constant", 0))
        len_heavy = len(labels_heavy_1)
        len_light = len(labels_light_1)

        return residue_embedding, labels_padded_1, len_heavy, len_light, pdb_code, numbers_heavy, numbers_light


class AminoAcidGraphSimple(Dataset):
    def __init__(self, dataset_dict: Dict, residue_embeddings: torch.Tensor, csv: pd.DataFrame, alpha: str = "4.5"):
        self.dataset_dict = dataset_dict
        self.residue_embeddings = residue_embeddings
        self.alpha = alpha
        self.csv = csv

    def __len__(self):
        return self.residue_embeddings.shape[0]

    def __getitem__(self, index):
        pdb = self.dataset_dict[str(index)]["pdb_code"]
        pdb_path = f"/home/gathenes/all_structures/imgt/{pdb}.pdb"
        heavy_chain = self.csv.query("pdb == @pdb")["Hchain"].values[0]
        light_chain = self.csv.query("pdb == @pdb")["Lchain"].values[0]
        chains = [heavy_chain, light_chain]

        # Filter the PDB data
        df_pdb = format_pdb(pdb_path).query("Atom_Name == 'CA' and Chain.isin(@chains)")
        # Separate heavy and light chains
        heavy_df = df_pdb.query("Chain == @heavy_chain")
        light_df = df_pdb.query("Chain == @light_chain")
        heavy_res_dict = {res_num: idx for idx, res_num in enumerate(heavy_df["Res_Num"])}
        light_res_dict = {res_num: idx for idx, res_num in enumerate(light_df["Res_Num"])}

        df_pdb = pd.concat([heavy_df, light_df])
        df_pdb["IMGT"] = df_pdb["Res_Num"].str.replace(r'[a-zA-Z]$', '', regex=True).astype(int)
        cdrs = list(range(25, 40 + 1)) + list(range(54, 67 + 1)) + list(range(103, 119 + 1))
        df_pdb=df_pdb.query("IMGT in @cdrs")

        # Prepare labels and embeddings
        labels_heavy = torch.tensor(self.dataset_dict[str(index)][f"H_id labels {self.alpha}"], dtype=torch.float32)
        labels_light = torch.tensor(self.dataset_dict[str(index)][f"L_id labels {self.alpha}"], dtype=torch.float32)
        embedding = self.residue_embeddings[index]

        # Collect features and labels for the graph nodes
        node_features = []
        node_labels = []

        # Heavy chain nodes
        for i, res in enumerate(heavy_df["Res_Num"].tolist()):
            res_index = heavy_res_dict[res]
            node_features.append(embedding[1 + res_index][2048:])  # Assuming embeddings are (num_nodes, 2048)
            node_labels.append(labels_heavy[res_index])

        # Light chain nodes
        for i, res in enumerate(light_df["Res_Num"].tolist()):
            res_index = light_res_dict[res]
            node_features.append(embedding[len(labels_heavy) + 2 + res_index][2048:])
            node_labels.append(labels_light[res_index])

        # Convert features and labels to tensors
        x = torch.stack(node_features)  # Shape: (num_nodes, 1024)
        y = torch.stack(node_labels)  # Shape: (num_nodes, 3)

        # Create edges based on 3D distances
        antibody_coords = df_pdb[["x", "y", "z"]].astype(float).values
        distances = np.linalg.norm(antibody_coords[:, np.newaxis] - antibody_coords, axis=2)
        antibody_indices, neighbor_indices = np.where((distances < 10) & (distances > 0))
        edges = [(i, j) for i, j in zip(antibody_indices, neighbor_indices)]

        # Prepare edge index for PyTorch Geometric
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Shape: (2, num_edges)

        # Return graph data and labels
        torch_graph = Data(x=x, edge_index=edge_index, y=y)
        return torch_graph


class AminoAcidGraphEGNN(Dataset):
    def __init__(self, dataset_dict: Dict, residue_embeddings: torch.Tensor, csv: pd.DataFrame, alpha: str = "4.5"):
        self.dataset_dict = dataset_dict
        self.residue_embeddings = residue_embeddings
        self.alpha = alpha
        self.csv = csv

    def __len__(self):
        return self.residue_embeddings.shape[0]

    def __getitem__(self, index):
        # Load pdb code and chains
        pdb = self.dataset_dict[str(index)]["pdb_code"]
        pdb_path = f"/home/gathenes/all_structures/imgt/{pdb}.pdb"
        heavy_chain = self.csv.query("pdb == @pdb")["Hchain"].values[0]
        light_chain = self.csv.query("pdb == @pdb")["Lchain"].values[0]
        chains = [heavy_chain, light_chain]

        # Filter the PDB data
        df_pdb = format_pdb(pdb_path).query("Atom_Name == 'CA' and Chain.isin(@chains)")
        df_pdb["IMGT"] = df_pdb["Res_Num"].str.replace(r'[a-zA-Z]$', '', regex=True).astype(int)
        # Separate heavy and light chains
        heavy_df = df_pdb.query("Chain == @heavy_chain")
        light_df = df_pdb.query("Chain == @light_chain")
        heavy_res_dict = {res_num: idx for idx, res_num in enumerate(heavy_df["Res_Num"])}
        light_res_dict = {res_num: idx for idx, res_num in enumerate(light_df["Res_Num"])}

        cdrs = list(range(25, 40 + 1)) + list(range(54, 67 + 1)) + list(range(103, 119 + 1))
        heavy_cdr = heavy_df.query("IMGT in @cdrs")
        light_cdr = light_df.query("IMGT in @cdrs")

        # Prepare labels and embeddings
        labels_heavy = torch.tensor(self.dataset_dict[str(index)][f"H_id labels {self.alpha}"], dtype=torch.float32)
        labels_light = torch.tensor(self.dataset_dict[str(index)][f"L_id labels {self.alpha}"], dtype=torch.float32)
        embedding = self.residue_embeddings[index]

        # Collect features (feats) and coordinates (coors)
        node_features = []
        node_coords = []
        node_labels = []
        # Heavy chain features and coordinates
        for i, res in enumerate(heavy_cdr["Res_Num"].tolist()):
            res_index = heavy_res_dict[res]
            node_features.append(embedding[1 + res_index][2048:])  # Assuming embeddings are (num_nodes, 2048)
            node_coords.append(heavy_cdr.iloc[i][["x", "y", "z"]].astype(float).values)
            node_labels.append(labels_heavy[res_index])

        # Light chain features and coordinates
        for i, res in enumerate(light_cdr["Res_Num"].tolist()):
            res_index = light_res_dict[res]
            node_features.append(embedding[len(labels_heavy) + 2 + res_index][2048:])
            node_coords.append(light_cdr.iloc[i][["x", "y", "z"]].astype(float).values)
            node_labels.append(labels_light[res_index])

        # Convert features and coordinates to tensors and reshape as required
        feats = torch.stack(node_features)  # Shape: (1, num_samples, num_feats)
        labels = torch.stack(node_labels)
        node_coords=np.array(node_coords)
        coors = torch.tensor(node_coords, dtype=torch.float32)  # Shape: (1, num_samples, 3)

        # Calculate pairwise distances for edges
        antibody_coords = np.array(node_coords)
        distances = np.linalg.norm(antibody_coords[:, np.newaxis] - antibody_coords, axis=2)

        # Generate the edges tensor as specified
        edges = torch.tensor(distances, dtype=torch.float32).unsqueeze(-1)  # Shape: (1, num_samples, num_samples, 1)

        return (feats, coors, edges),labels
