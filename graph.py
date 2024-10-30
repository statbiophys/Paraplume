from typing import Dict, Tuple

import networkx as nx
import numpy as np
import torch
from Bio.PDB import PDBParser
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils import format_pdb


# Load PDB and extract C-alpha coordinates
def extract_ca_coordinates(pdb_path: str) -> Tuple[np.ndarray, list]:
    df_pdb = format_pdb(pdb_path)
    ca_coords = []
    residues = []

    for model in df_pdb:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords.append(residue["CA"].coord)
                    residues.append(residue)

    ca_coords = np.array(ca_coords)
    return ca_coords, residues

# Graph creation class
class AminoAcidGraph(Dataset):
    def __init__(self, pdb_path: str, dataset_dict: Dict, residue_embeddings: torch.Tensor, alpha: str = "4.5"):
        self.pdb_path = pdb_path
        self.dataset_dict = dataset_dict
        self.residue_embeddings = residue_embeddings
        self.alpha = alpha
        self.ca_coords, self.residues = extract_ca_coordinates(pdb_path)

    def __len__(self):
        return len(self.residue_embeddings)

    def __getitem__(self, index):
        labels_heavy_1 = self.dataset_dict[str(index)][f"H_id labels {self.alpha}"]
        labels_light_1 = self.dataset_dict[str(index)][f"L_id labels {self.alpha}"]

        residue_embedding = self.residue_embeddings[index, :, :]
        labels_paired_1 = labels_heavy_1 + labels_light_1
        labels_padded_1 = torch.FloatTensor(
            torch.nn.functional.pad(
                torch.FloatTensor(labels_paired_1),
                (0, 280 - len(torch.FloatTensor(labels_paired_1))),
                "constant",
                0,
            )
        )
        len_heavy = len(labels_heavy_1)
        len_light = len(labels_light_1)

        # Create the graph
        graph_data = self.create_graph(index, residue_embedding)

        return graph_data, labels_padded_1, len_heavy, len_light

    def create_graph(self, index, residue_embedding) -> Data:
        graph = nx.Graph()

        # Add nodes with embeddings as features
        for i, embedding in enumerate(residue_embedding):
            graph.add_node(i, x=embedding)
        # Compute distances and add edges for nodes within 10 Å
        for i in range(len(self.ca_coords)):
            for j in range(i + 1, len(self.ca_coords)):
                distance = np.linalg.norm(self.ca_coords[i] - self.ca_coords[j])
                if distance <= 10.0:
                    graph.add_edge(i, j)

        # Convert to PyTorch Geometric Data object
        edge_index = torch.tensor(list(graph.edges)).t().contiguous()
        x = torch.stack([graph.nodes[i]["x"] for i in graph.nodes])

        return Data(x=x, edge_index=edge_index)
