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
            if isinstance(idx_group,list):
                # Adjust indices (Python is 0-based)
                adjusted_indices = [i for i in idx_group]
                # Extract coordinates and calculate COM
                group_coords = np.array(data.pos[adjusted_indices])
                atom_types = data.z[adjusted_indices]
                mass = np.array([mass_mapping[int(type)] for type in atom_types])
                com = calculate_com(group_coords,mass)
                cg_positions.append(com)
            else:
                cg_positions.append(data.pos[idx_group].numpy())
        
        # Create new data instance
        cg_data = Data(pos=torch.tensor(np.array(cg_positions), dtype=torch.float),
                       z=torch.tensor(cg_types, dtype=torch.long),
                       homo=data.homo)
        new_dataset.append(cg_data)
        mc.write_xyz_file(cg_types, cg_positions, 'tmp.xyz')

    return new_dataset

def flatten_list(mixed_list):
    # Flatten the list by iterating and checking for sublists
    return [item for elem in mixed_list for item in (elem if isinstance(elem, list) else [elem])]

def find_CG_mapping(mol):
    idx_groups = []
    type_groups = []
    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 16:  # Check if the atom is sulfur (S)
            # print(f"Sulfur atom found, index: {atom.GetIdx()}")
            # idx_tmp.append(atom.GetIdx())
            idx_groups.append(atom.GetIdx())
            type_groups.append(1)
            # Get neighbors of the sulfur atom
            neighbors = atom.GetNeighbors()
            
            # Iterate over the neighboring atoms and print their details
            for neighbor in neighbors:
                # print(f" - Neighbor: Atom index {neighbor.GetIdx()}, Element: {neighbor.GetSymbol()}, Bond type: {mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()}")
                idx_groups.append(neighbor.GetIdx())
                type_groups.append(2)

    for atom in mol.GetAtoms():
        if atom.GetIdx() in flatten_list(idx_groups):
            continue
        if atom.IsInRing():
            idx_tmp = []
            idx_tmp.append(atom.GetIdx())
            # print(f"ring atom found, index: {atom.GetIdx()} atomic_number: {atom.GetAtomicNum()}")
            neighbors = atom.GetNeighbors()
            nb_no_ring = 0
            for neighbor in neighbors:
                if not neighbor.IsInRing():
                    nb_no_ring+=1
                    # print(f" - Neighbor: Atom index {neighbor.GetIdx()}, Element: {neighbor.GetSymbol()}, Bond type: {mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()}")
                    idx_tmp.append(neighbor.GetIdx())
                    neighbor_tmp = neighbor
            if nb_no_ring == 0: # -H
                idx_groups.append(idx_tmp)
                type_groups.append(3) 
            else:
                if neighbor_tmp.GetAtomicNum() == 9:
                    idx_groups.append(idx_tmp) # F
                    type_groups.append(6) # 6+9

                if neighbor_tmp.GetAtomicNum() == 8:
                    idx_groups.append(idx_tmp) # O
                    type_groups.append(5) # 6+8+6

                if neighbor_tmp.GetAtomicNum() == 6:
                    neighbors2 = neighbor_tmp.GetNeighbors()
                    neighbor_tmp2 = [neighbor2 for neighbor2 in neighbors2 if not neighbor2.IsInRing()]
                    if len(neighbor_tmp2)==0:
                        idx_groups.append(idx_tmp) # CH3
                        type_groups.append(4) 
                    elif neighbor_tmp2[0].GetAtomicNum()==7:
                        idx_tmp.append(neighbor_tmp2[0].GetIdx())
                        idx_groups.append(idx_tmp) # -C-N
                        type_groups.append(7)
                    else:
                        print('error')

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
torch.save(cg_data_all, './data_CG5COM.pt')




