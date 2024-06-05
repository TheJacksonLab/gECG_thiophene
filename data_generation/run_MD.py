import os 
import numpy as np
import tools_lammps as tl

def read_trj(new_dir):
    frame, t_list = tl.read_lammps_dump_xyz('./dump_relax.xyz')

    lmp = tl.read_lammps_full('./out_nvt.dat')

    mass2type = {'1.008':'H',
                '12.011':'C',
                '32.06':'S',
                '18.998':'F',
                '14.007':'N',
                '15.999':'O'}

    atom_type = []
    for i,each_atom in enumerate(lmp.mass):
        atom_type.append(mass2type[each_atom[1]])
    natoms = len(atom_type)
    for itime in range(len(t_list)):
        os.system('mkdir {}/str_{}'.format(new_dir,itime))
        with open('{}/str_{}/tmp.xyz'.format(new_dir,itime),'w') as f:
            f.write('{}\n'.format(natoms))
            f.write('\n')
            for iatom in range(natoms):
                f.write('{0:s} {1:.6f} {2:.6f} {3:.6f}\n'.format(atom_type[iatom],frame[itime]['x'][iatom],
                                            frame[itime]['y'][iatom],
                                            frame[itime]['z'][iatom]))
        os.system('cp lstr.inp {}/str_{}/'.format(new_dir,itime))


for i in range(0,1000):
    if os.path.isfile(f'tp{i}.lmp'):
        try: 
            os.system(f'cp tp{i}.lmp tp.lmp')
            os.system('lmp_serial -in in.nvt1')
            os.system('lmp_serial -in in.nvt2')
            os.system(f'mkdir tp{i}')
            read_trj(f'tp{i}')
        except:
            os.system(f'rm -r tp{i}')
            continue

        

