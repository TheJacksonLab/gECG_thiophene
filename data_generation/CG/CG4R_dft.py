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
    adjusted_indices = [i for i in idx_groups]
    for data in dataset:
        # Extract coordinates and calculate COM
        group_coords = data.pos[adjusted_indices]
        
        # Create new data instance
        cg_data = Data(pos=group_coords,
                       z=torch.tensor(cg_types, dtype=torch.long),
                       homo=data.homo,
                       homo1to9=data.homo1to9, 
                       lumo=data.lumo,
                       lumo1=data.lumo1,
                       y=data.homo)
        new_dataset.append(cg_data)
        mc.write_xyz_file(cg_types, group_coords, 'tmp.xyz')
    return new_dataset

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
        if atom.GetIdx() in idx_groups:
            continue
        if atom.IsInRing():
            # print(f"ring atom found, index: {atom.GetIdx()} atomic_number: {atom.GetAtomicNum()}")
            neighbors = atom.GetNeighbors()
            nb_no_ring = 0
            for neighbor in neighbors:
                if not neighbor.IsInRing():
                    nb_no_ring+=1
                    # print(f" - Neighbor: Atom index {neighbor.GetIdx()}, Element: {neighbor.GetSymbol()}, Bond type: {mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()}")
                    # idx_tmp.append(neighbor.GetIdx())
                    neighbor_tmp = neighbor
            if nb_no_ring == 0: # -H
                idx_groups.append(atom.GetIdx())
                type_groups.append(3) 
            else:
                if neighbor_tmp.GetAtomicNum() == 9:
                    idx_groups.append(atom.GetIdx()) # F
                    type_groups.append(6) # 6+9

                if neighbor_tmp.GetAtomicNum() == 8:
                    idx_groups.append(atom.GetIdx()) # O
                    type_groups.append(5) # 6+8+6

                if neighbor_tmp.GetAtomicNum() == 6:
                    neighbors2 = neighbor_tmp.GetNeighbors()
                    neighbor_tmp2 = [neighbor2 for neighbor2 in neighbors2 if not neighbor2.IsInRing()]
                    if len(neighbor_tmp2)==0:
                        idx_groups.append(atom.GetIdx()) # CH3
                        type_groups.append(4) 
                    elif neighbor_tmp2[0].GetAtomicNum()==7:
                        # idx_tmp.append(neighbor_tmp2[0].GetIdx())
                        idx_groups.append(atom.GetIdx()) # -C-N
                        type_groups.append(7)
                    else:
                        print('error')

    # remove S
    type_groups_f, idx_groups_f = zip(*[(x, y) for x, y in zip(type_groups, idx_groups) if x != 1])

    # Converting from tuples back to lists, if needed
    idx_groups_f = list(idx_groups_f)
    type_groups_f = list(type_groups_f)
        
    return idx_groups_f, type_groups_f

mass_mapping = {1:1.008,6:12.011,7:14.007,8:15.999,16:32.065,9:18.998}

cg_dataset_list = []
for i in range(1000):
    if os.path.isdir(f'dft/tp{i}'):
        # Load a molecule from a .mol or .pdb file
        mol_file = f'tp{i}.mol'  # Or .pdb file
        mol = Chem.MolFromMolFile(mol_file)  # Use MolFromPDBFile for .pdb files

        dataset = torch.load(f'./dft/tp{i}/dftdata.pt')
        idx_groups, cg_types = find_CG_mapping(mol)
        # print(np.sort(np.concatenate(idx_groups)))
        cg_dataset = coarse_grain_dataset(dataset, idx_groups, cg_types)
        cg_dataset_list.append(cg_dataset)

cg_data_all = ConcatDataset(cg_dataset_list)
torch.save(cg_data_all, './dftdata_CG4R.pt')