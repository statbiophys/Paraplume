from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_node_embedding, read_pdb_to_dataframe


class AminoAcidGraphEGNN(Dataset):
    def __init__(
        self,
        dataset_dict: Dict,
        embeddings: torch.Tensor,
        csv: pd.DataFrame,
        alpha: str = "4.5",
        pdb_folder_path="/home/gathenes/paragraph_benchmark/abb3_pdbs_renumbered",
        embedding_models="one-hot",
        graph_distance: float = 10,
        region: str = "cdrs",
    ):
        self.dataset_dict = dataset_dict
        self.embeddings = embeddings
        self.alpha = alpha
        self.csv = csv
        self.pdb_folder_path = pdb_folder_path
        self.embedding_models = embedding_models
        self.graph_distance = graph_distance
        self.region = region

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, index):
        # Load pdb code and chains
        pdb = self.dataset_dict[str(index)]["pdb_code"]
        pdb_path = f"{self.pdb_folder_path}/{pdb}.pdb"
        heavy_chain = self.csv.query("pdb == @pdb")["Hchain"].values[0]
        light_chain = self.csv.query("pdb == @pdb")["Lchain"].values[0]
        chains = [heavy_chain, light_chain]
        # Filter the PB data
        df_pdb = read_pdb_to_dataframe(pdb_path).query(
            "atom_name == 'CA' and chain_id.isin(@chains)"
        )
        df_pdb["residue_letter"] = df_pdb["residue_name"].map(triplet_to_one_letter)

        # Separate heavy and light chains
        heavy_df = df_pdb.query("chain_id == @heavy_chain")
        light_df = df_pdb.query("chain_id == @light_chain")
        heavy_res_dict = {res_num: idx for idx, res_num in enumerate(heavy_df["IMGT"])}
        light_res_dict = {res_num: idx for idx, res_num in enumerate(light_df["IMGT"])}
        if self.region == "cdrs":
            region = list(range(25, 40 + 1)) + list(range(54, 67 + 1)) + list(range(103, 119 + 1))
        elif self.region == "fv":
            region = list(range(129))
        else:
            raise ValueError("Invalid region. Choose 'cdrs' or 'fv'.")
        heavy_cdr = heavy_df.query("residue_number in @region")
        light_cdr = light_df.query("residue_number in @region and residue_number<128")

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
        imgt_numbers_heavy = []
        # Heavy chain features and coordinates
        for i, (res, name) in enumerate(
            zip(heavy_cdr["IMGT"].tolist(), heavy_cdr["residue_letter"].tolist())
        ):
            imgt_numbers_heavy.append(res)
            res_index = heavy_res_dict[res]
            if self.embedding_models == ["one-hot"]:
                node_feature = torch.tensor([int(each == name) for each in amino_acids] + [1, 0])
                node_features.append(node_feature)
            else:
                new_embedding = get_node_embedding(
                    embedding,
                    index=res_index,
                    embedding_models=self.embedding_models,
                    chain="heavy",
                )
                node_features.append(new_embedding)
            node_coords.append(
                heavy_cdr.iloc[i][["x_coord", "y_coord", "z_coord"]].astype(float).values
            )
            node_labels.append(labels_heavy[res_index])

        # Light chain features and coordinates
        imgt_numbers_light = []
        for i, (res, name) in enumerate(
            zip(light_cdr["IMGT"].tolist(), light_cdr["residue_letter"].tolist())
        ):
            imgt_numbers_light.append(res)
            res_index = light_res_dict[res]

            if self.embedding_models == ["one-hot"]:
                node_feature = torch.tensor([int(each == name) for each in amino_acids] + [0, 1])
                node_features.append(node_feature)
            else:
                new_embedding = get_node_embedding(
                    embedding,
                    index=len(labels_heavy) + res_index,
                    embedding_models=self.embedding_models,
                    chain="light",
                )
                node_features.append(new_embedding)
            node_coords.append(
                light_cdr.iloc[i][["x_coord", "y_coord", "z_coord"]].astype(float).values
            )
            node_labels.append(labels_light[res_index])

        # Convert features and coordinates to tensors and reshape as required
        feats = torch.stack(node_features)  # Shape: (1, num_samples, num_feats)
        labels = torch.stack(node_labels)
        node_coords = np.array(node_coords)
        coors = torch.tensor(node_coords, dtype=torch.float32)  # Shape: (1, num_samples, 3)

        # Calculate pairwise distances for edges
        antibody_coords = np.array(node_coords)
        distances = np.linalg.norm(antibody_coords[:, np.newaxis] - antibody_coords, axis=2)

        # Generate the edges tensor as specified
        edges = torch.tensor(distances, dtype=torch.float32).unsqueeze(
            -1
        )  # Shape: (1, num_samples, num_samples, 1)
        edges = (edges < self.graph_distance).int()
        num_samples = edges.shape[0]
        edges[torch.arange(num_samples), torch.arange(num_samples), 0] = 0

        return (
            (feats.to(torch.float), coors.to(torch.float), edges.to(torch.float)),
            labels.to(torch.float),
            pdb,
            imgt_numbers_heavy,
            imgt_numbers_light,
        )

def create_graph_dataloader(
    pdb_folder_path: Path,
    dataset_dict: Dict,
    embeddings: torch.Tensor,
    csv,
    shuffle: bool = False,
    embedding_models="one-hot",
    graph_distance: float = 10,
    region: str = "cdrs",
    num_workers: int = 1,
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
        embedding_models=embedding_models,
        graph_distance=graph_distance,
        region=region,
    )
    dataset_loader = DataLoader(dataset, shuffle=shuffle, num_workers=num_workers)
    return dataset_loader



triplet_to_one_letter = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

amino_acids = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
