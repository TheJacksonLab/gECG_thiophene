import numpy as np
import my_common as mc
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import os

atom_mapping = {
    'H': 1,  # Hydrogen
    'O': 8,  # Oxygen
    'C': 6,  # Carbon
    'F': 9,  
    'S': 16, 
    'N': 7,
    # Add other atoms as needed
}

total_data = []
for itp in range(1000):
    dir_name = f'dft_L10/tp{itp}' # might need to change
    if os.path.isdir(dir_name):
        # os.system(f'rm {dir_name}/dft.e* {dir_name}/dft.o* {dir_name}/orca.o*')
        dataset = []

        no_err=True
        try:
            e_orbitals = mc.read_ORCA('./{}/log'.format(dir_name),search_words='ORBITAL ENERGIES',skip_lines=3,stop_words='*')
            # convert to array of floats
            e_orbitals = np.array(e_orbitals[:-1].tolist(), dtype=float)
            HOMO = e_orbitals[e_orbitals[:,1]!=0][-1,-1]
            HOMO1 = e_orbitals[e_orbitals[:,1]!=0][-2,-1]
            HOMO2 = e_orbitals[e_orbitals[:,1]!=0][-3,-1]
            HOMO3 = e_orbitals[e_orbitals[:,1]!=0][-4,-1]
            HOMO4 = e_orbitals[e_orbitals[:,1]!=0][-5,-1]
            HOMO5 = e_orbitals[e_orbitals[:,1]!=0][-6,-1]
            HOMO6 = e_orbitals[e_orbitals[:,1]!=0][-7,-1]
            HOMO7 = e_orbitals[e_orbitals[:,1]!=0][-8,-1]
            HOMO8 = e_orbitals[e_orbitals[:,1]!=0][-9,-1]
            HOMO9 = e_orbitals[e_orbitals[:,1]!=0][-10,-1]
            LUMO = e_orbitals[e_orbitals[:,1]==0][0,-1]
            LUMO1 = e_orbitals[e_orbitals[:,1]==0][1,-1]
            df = pd.read_csv(f'./{dir_name}/tmp.xyz', delim_whitespace=True, skiprows=2, names=['atom_type', 'x', 'y', 'z'])
            atom_type = df['atom_type'].values
            atom_type = [atom_mapping[atom] for atom in atom_type]
            new_instance = Data(homo=HOMO,
                                homo1to9=[HOMO1,HOMO2,HOMO3,HOMO4,HOMO5,HOMO6,HOMO7,HOMO8,HOMO9],
                                lumo=LUMO,
                                lumo1=LUMO1,
                                pos=torch.Tensor(df.loc[:,'x':].to_numpy()),z=torch.Tensor(atom_type))
            dataset.append(new_instance)
            total_data.append(new_instance)
        except:
            print(f'error in tp{itp}')
            no_err = False
            os.system('mv ./{} ./f{}'.format(dir_name,dir_name))
            break

        if no_err:        
            torch.save(dataset, f'{dir_name}/dftdata.pt')
torch.save(total_data, 'dftdata_AA.pt')