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
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 16:  # Check if the atom is sulfur (S)
            # print(f"Sulfur atom found, index: {atom.GetIdx()}")
            idx_tmp = []
            idx_tmp.append(atom.GetIdx())
            # Get neighbors of the sulfur atom
            neighbors = atom.GetNeighbors()
            
            # Iterate over the neighboring atoms and print their details
            for neighbor in neighbors:
                # print(f" - Neighbor: Atom index {neighbor.GetIdx()}, Element: {neighbor.GetSymbol()}, Bond type: {mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()}")
                idx_tmp.append(neighbor.GetIdx())
            idx_groups.append(idx_tmp)
            type_groups.append(16+6*2)
    
    def add_neighbor(neighbors, idx_tmp, bead_type, idx_groups, depth):
        if depth > 2:  # Limit recursion depth to 3 (0, 1, 2)
            return idx_tmp, bead_type

        for neighbor in neighbors:
            neighbor_idx = neighbor.GetIdx()
            # Check if the neighbor is already included
            if neighbor_idx not in idx_tmp and all(neighbor_idx not in g for g in idx_groups):
                idx_tmp.append(neighbor_idx)
                bead_type += neighbor.GetAtomicNum()
                # Recursive call with incremented depth
                idx_tmp, bead_type = add_neighbor(neighbor.GetNeighbors(), idx_tmp, bead_type, idx_groups, depth + 1)
        return idx_tmp, bead_type

    for atom in mol.GetAtoms():
        if atom.GetIdx() in list(np.concatenate(idx_groups)):
            continue
        if atom.IsInRing():
            idx_tmp = []
            idx_tmp.append(atom.GetIdx())
            bead_type = atom.GetAtomicNum()
            # print(f"ring atom found, index: {atom.GetIdx()} atomic_number: {atom.GetAtomicNum()}")
            neighbors = atom.GetNeighbors()
            idx_tmp,bead_type = add_neighbor(neighbors,idx_tmp,bead_type,idx_groups,depth=0)
            idx_groups.append(idx_tmp)
            type_groups.append(bead_type)
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
torch.save(cg_data_all, './data_CG2.pt')
