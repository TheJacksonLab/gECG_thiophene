from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
import numpy as np
import os
from tqdm import tqdm

thiophene_H_H    = 'c1ccsc1'
thiophene_H_CH3  = 'c1c(C)csc1'
thiophene_H_OCH3 = 'c1c(OC)csc1'
thiophene_H_F    = 'c1c(F)csc1'
thiophene_H_CN   = 'c1c(C#N)csc1'

thiophene_CH3_H    = 'Cc1ccsc1'
thiophene_CH3_CH3  = 'Cc1c(C)csc1'
thiophene_CH3_OCH3 = 'Cc1c(OC)csc1'
thiophene_CH3_F    = 'Cc1c(F)csc1'
thiophene_CH3_CN   = 'Cc1c(C#N)csc1'

thiophene_OCH3_H = 'COc1ccsc1'
thiophene_OCH3_CH3= 'COc1c(C)csc1'
thiophene_OCH3_OCH3 = 'COc1c(OC)csc1'
thiophene_OCH3_F = 'COc1c(F)csc1'
thiophene_OCH3_CN = 'COc1c(C#N)csc1'

thiophene_F_H = 'Fc1ccsc1'
thiophene_F_CH3= 'Fc1c(C)csc1'
thiophene_F_OCH3 = 'Fc1c(OC)csc1'
thiophene_F_F = 'Fc1c(F)csc1'
thiophene_F_CN = 'Fc1c(C#N)csc1'

thiophene_CN_H = 'N#Cc1ccsc1'
thiophene_CN_CH3= 'N#Cc1c(C)csc1'
thiophene_CN_OCH3 = 'N#Cc1c(OC)csc1'
thiophene_CN_F = 'N#Cc1c(F)csc1'
thiophene_CN_CN = 'N#Cc1c(C#N)csc1'

monomer = [Chem.MolFromSmiles(thiophene_H_H    ),
           Chem.MolFromSmiles(thiophene_H_CH3 ),
           Chem.MolFromSmiles(thiophene_H_OCH3),
           Chem.MolFromSmiles(thiophene_H_F    ),
           Chem.MolFromSmiles(thiophene_H_CN   ), 

           Chem.MolFromSmiles(thiophene_CH3_H   ),
           Chem.MolFromSmiles(thiophene_CH3_CH3 ),
           Chem.MolFromSmiles(thiophene_CH3_OCH3),
           Chem.MolFromSmiles(thiophene_CH3_F   ),
           Chem.MolFromSmiles(thiophene_CH3_CN  ), 

           Chem.MolFromSmiles(thiophene_OCH3_H    ),
           Chem.MolFromSmiles(thiophene_OCH3_CH3  ),
           Chem.MolFromSmiles(thiophene_OCH3_OCH3 ),
           Chem.MolFromSmiles(thiophene_OCH3_F    ),
           Chem.MolFromSmiles(thiophene_OCH3_CN   ),

           Chem.MolFromSmiles(thiophene_F_H    ),
           Chem.MolFromSmiles(thiophene_F_CH3  ),
           Chem.MolFromSmiles(thiophene_F_OCH3 ),
           Chem.MolFromSmiles(thiophene_F_F    ),
           Chem.MolFromSmiles(thiophene_F_CN   ), 

           Chem.MolFromSmiles(thiophene_CN_H    ),
           Chem.MolFromSmiles(thiophene_CN_CH3  ),
           Chem.MolFromSmiles(thiophene_CN_OCH3 ),
           Chem.MolFromSmiles(thiophene_CN_F    ),
           Chem.MolFromSmiles(thiophene_CN_CN   ), 
           ]

def combine_monomers(monomer_list):
    combined_mol = monomer_list[0]
    for i in monomer_list[1:]:
        combined_mol = Chem.CombineMols(combined_mol, i)
    return combined_mol

lengths_list = np.arange(5,16,1)

os.system('mkdir tp_gen')
smile_list = []
idx_list = []
for i in tqdm(range(200)):
    deg_pol = np.random.choice(lengths_list)
    print(deg_pol)
    random_idx = np.random.randint(0,len(monomer),deg_pol)
    # if i>1:
    #     while all(all(random_idx == sublist) for sublist in idx_list) or all(all(random_idx[::-1] == sublist) for sublist in idx_list):
    #         random_idx = np.random.randint(0,len(monomer),deg_pol)
    idx_list.append(random_idx)
    combined_mol = combine_monomers([monomer[random_idx[i]] for i in range(deg_pol)])
    edcombo = Chem.EditableMol(combined_mol)
    num_atoms = np.array([monomer[random_idx[i]].GetNumAtoms() for i in range(deg_pol)])

    # polymerization
    for j in range(1,deg_pol):
        if j%2 == 1:
            edcombo.AddBond(int(np.sum(num_atoms[:j])-1),int(np.sum(num_atoms[:(j+1)])-1),order=Chem.rdchem.BondType.SINGLE)
        else:
            edcombo.AddBond(int(np.sum(num_atoms[:j])-3),int(np.sum(num_atoms[:(j+1)])-3),order=Chem.rdchem.BondType.SINGLE)

    back = edcombo.GetMol()
    smile = Chem.MolToSmiles(back)
    smile_list.append(smile)
    if i<100:
        Draw.MolToFile(back, f'tp_gen/tp{i}.png',size=[400,200])

with open('smile.txt',"w") as file:
    for i in smile_list:
        file.write(i)
        file.write('\n')

with open('idx.txt',"w") as file:
    for i in idx_list:
        i = i+1 # starting from 1, 0 represents no atom
        # total_padding = 15-deg_pol
        # padding_start = np.random.randint(0, total_padding + 1)
        # padding_end = total_padding - padding_start
        # padded_array = np.pad(i, (padding_start, padding_end), 'constant', constant_values=(0))
        # padded_array = np.pad(i, (0, 15-deg_pol), 'constant', constant_values=(0))
        array_str = np.array2string(i, separator=', ')[1:-1]
        file.write(array_str)
        file.write('\n')
