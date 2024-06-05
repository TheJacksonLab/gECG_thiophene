import torch
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from torch.utils.data import ConcatDataset
import my_common as mc
def calculate_com(coordinates,mass):
    # Calculate the center of mass (mean of coordinates)
    return np.sum(coordinates*mass[:, np.newaxis], axis=0)/np.sum(mass)

def coarse_grain_dataset(dataset, idx_groups, cg_types):
    new_dataset = []
    for data in dataset:
        cg_positions = []
        for idx_group in idx_groups:
            # Adjust indices (Python is 0-based)
            adjusted_indices = [i for i in idx_group]
            # Extract coordinates and calculate COM
            group_coords = np.array(data.pos[adjusted_indices])
            atom_types = data.z[adjusted_indices]
            mass = np.array([mass_mapping[int(type)] for type in atom_types])
            com = calculate_com(group_coords,mass)
            cg_positions.append(com)
        
        # Create new data instance
        cg_data = Data(pos=torch.tensor(np.array(cg_positions), dtype=torch.float),
                       z=torch.tensor(cg_types, dtype=torch.long),
                       homo=data.homo)
        new_dataset.append(cg_data)
        mc.write_xyz_file(cg_types, cg_positions, 'tmp.xyz')

    return new_dataset

def find_CG_mapping(mol):
    idx_groups = []
    type_groups = []
    # Iterate over all atoms in the molecule
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        # print("Ring atoms: ", ring)
        idx_tmp = list(ring)
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            # print("Atom symbol: ", atom.GetSymbol())
            neighbors = atom.GetNeighbors() 
            n_neighbor_S = np.sum([neighbor.GetAtomicNum() == 16 for neighbor in neighbors])
            if n_neighbor_S>0:
                continue
            else: 
                for neighbor in neighbors:
                    if not neighbor.IsInRing():
                        idx_tmp.append(neighbor.GetIdx())
                        neighbors2 = neighbor.GetNeighbors()
                        neighbor_tmp2 = [neighbor2 for neighbor2 in neighbors2 if not neighbor2.IsInRing()]
                        if len(neighbor_tmp2)==1:
                            idx_tmp.append(neighbor_tmp2[0].GetIdx())
                        elif len(neighbor_tmp2)>1:
                            print(f'{neighbor.GetAtomicNum()} {neighbor_tmp2.GetAtomicNum()}')
        idx_groups.append(idx_tmp)
        type_groups.append(np.sum([mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in idx_tmp]))

    return idx_groups, type_groups

mass_mapping = {1:1.008,6:12.011,7:14.007,8:15.999,16:32.065,9:18.998}

cg_dataset_list = []
for i in range(1000):
    if os.path.isdir(f'tp{i}'):
        # Load a molecule from a .mol or .pdb file
        mol_file = f'tp{i}.mol'  # Or .pdb file
        mol = Chem.MolFromMolFile(mol_file)  # Use MolFromPDBFile for .pdb files

        dataset = torch.load(f'./tp{i}/data.pt')
        idx_groups, cg_types = find_CG_mapping(mol)
        # print(np.sort(np.concatenate(idx_groups)))
        cg_dataset = coarse_grain_dataset(dataset, idx_groups, cg_types)
        cg_dataset_list.append(cg_dataset)

cg_data_all = ConcatDataset(cg_dataset_list)
torch.save(cg_data_all, './data_CG1.pt')
