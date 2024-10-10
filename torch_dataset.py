
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

    def __init__(self, dataset_dict: Dict, residue_embeddings:torch.Tensor, max_length:int):
        self.dataset_dict = dataset_dict
        self.residue_embeddings = residue_embeddings
        self.max_length=max_length

    def __len__(self):
        return self.residue_embeddings.shape[0]

    def __getitem__(self, index):
        labels_heavy, numbers_heavy = self.dataset_dict[str(index)]["H_id labels"], self.dataset_dict[str(index)]["H_id numbers"]
        labels_light, numbers_light = self.dataset_dict[str(index)]["L_id labels"], self.dataset_dict[str(index)]["L_id numbers"]

        residue_embedding = self.residue_embeddings[index,:,:]

        labels_paired = labels_heavy+labels_light
        len_labels_paired = len(labels_paired)
        labels_padded = torch.FloatTensor(F.pad(torch.FloatTensor(labels_paired), (0, self.max_length-len(torch.FloatTensor(labels_paired))),"constant", 0))

        inverse_number_heavy = {each : i for i,each in enumerate(numbers_heavy)}
        inverse_number_light = {each : i for i,each in enumerate(numbers_light)}
        heavy_indices, light_indices = get_cdr_pm2_indices(inverse_number_heavy, inverse_number_light)
        heavy_indices = torch.FloatTensor(F.pad(torch.FloatTensor(heavy_indices), (0, 100-len(torch.FloatTensor(heavy_indices))),"constant", -1))
        light_indices = torch.FloatTensor(F.pad(torch.FloatTensor(light_indices), (0, 100-len(torch.FloatTensor(light_indices))),"constant", -1))

        len_padd_heavy = len(heavy_indices)
        len_padd_light = len(light_indices)
        len_heavy = len(labels_heavy)

        return residue_embedding, labels_padded, len_labels_paired, heavy_indices, light_indices, len_padd_heavy, len_padd_light, len_heavy
