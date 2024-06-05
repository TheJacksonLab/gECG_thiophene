import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from torch.utils.data import ConcatDataset

cg_dataset_list = []
for i in range(1000):
    if os.path.isdir(f'tp{i}'):

        dataset = torch.load(f'./tp{i}/data.pt')
        new_dataset = []
        for data in dataset:
            pos = data['pos'][data['z']!=1]
            z = data['z'][data['z']!=1]
            homo = data['homo']
            cg_data = Data(pos=pos,
                        z=z,
                        homo=homo,y=homo)
            new_dataset.append(cg_data)
        cg_dataset_list.append(new_dataset)

cg_data_all = ConcatDataset(cg_dataset_list)
torch.save(cg_data_all, './data_UA.pt')





