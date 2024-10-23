
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


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
        labels_heavy = self.dataset_dict[str(index)][f"H_id labels {self.alpha}"]
        labels_light = self.dataset_dict[str(index)][f"L_id labels {self.alpha}"]

        residue_embedding = self.residue_embeddings[index,:,:]

        labels_paired = labels_heavy+labels_light
        labels_padded = torch.FloatTensor(F.pad(torch.FloatTensor(labels_paired), (0, 259-len(torch.FloatTensor(labels_paired))),"constant", 0))
        len_heavy = len(labels_heavy)
        len_light = len(labels_light)

        return residue_embedding, labels_padded, len_heavy, len_light

class ParatopePredictDataset(Dataset):

    def __init__(self, dataset_dict: Dict, residue_embeddings:torch.Tensor, alpha:str="4.5"):
        self.dataset_dict = dataset_dict
        self.residue_embeddings = residue_embeddings
        self.alpha=alpha

    def __len__(self):
        return self.residue_embeddings.shape[0]

    def __getitem__(self, index):
        labels_heavy = self.dataset_dict[str(index)][f"H_id labels {self.alpha}"]
        labels_light = self.dataset_dict[str(index)][f"L_id labels {self.alpha}"]
        pdb_code = self.dataset_dict[str(index)]["pdb_code"]
        numbers_heavy = self.dataset_dict[str(index)]["H_id numbers"]
        numbers_light = self.dataset_dict[str(index)]["L_id numbers"]

        residue_embedding = self.residue_embeddings[index,:,:]

        labels_paired = labels_heavy+labels_light
        labels_padded = torch.FloatTensor(F.pad(torch.FloatTensor(labels_paired), (0, 259-len(torch.FloatTensor(labels_paired))),"constant", 0))
        len_heavy = len(labels_heavy)
        len_light = len(labels_light)

        return residue_embedding, labels_padded, len_heavy, len_light, pdb_code, numbers_heavy, numbers_light
