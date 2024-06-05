import os
import subprocess

smiles_list= []
with open('smile.txt','r') as file:
    for line in file:
        # Strip whitespace and newlines, then append if the line is not empty
        smiles = line.strip()
        if smiles:  # Check if the line is not empty
            smiles_list.append(smiles)

subprocess.run('rm ~/save_opls/tp*', shell=True)
for i in range(0, len(smiles_list)):
    if not os.path.isfile(f'./tp_gen/tp{i}.lmp'):
        print(f'Processing TP{i}: {smiles_list[i]}',flush=True)
        success = False
        attempts = 0
        while not success and attempts < 10:
            # Running LigParGen without stopping the script on failure
            subprocess.run(f'LigParGen -s \'{smiles_list[i]}\' -r tp{i} -c 0 -o 3', shell=True)
            
            # Attempt to copy the .lmp file
            result_lmp = subprocess.run(f'cp ~/save_opls/tp{i}.lmp ./tp_gen/', shell=True)
            
            # Attempt to copy the .mol file, assuming success doesn't depend on this operation
            subprocess.run(f'cp ~/save_opls/tp{i}.mol ./tp_gen/', shell=True)
            
            # Check if the copy operation for .lmp was successful by examining the returncode
            if result_lmp.returncode == 0:
                success = True
            else:
                print(f"Attempt {attempts + 1} failed for TP{i}. Retrying...",flush=True)
            
            attempts += 1  # Increment attempts after each loop iteration

        if not success:
            print(f"Failed to process TP{i} after {attempts} attempts.",flush=True)

# os.system('rm ~/save_opls/tp*')
# os.system('LigParGen -s \'{}\' -r tp{} -c 0 -o 3'.format(smile))

