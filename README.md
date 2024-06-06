# gECG_thiophene

Code for Generalized ECG (gECG) for thiophene polymers paper

## Introduction

Electronic coarse graining (ECG) is a machine learning approach that can predict quantum mechanical (QM) optoelectronic properties of molecules (and materials) directly based on molecular configurations at the coarse-grained (CG) resolution.

This project invovles the development of generalized ECG (gECG) models for thiophene polymers, including the necessary codes, datasets, and the trained models. More information can be found in our preprint paper ''Generalized Electronic Coarse Graining for Thiophene Polymers'' via XXX.

## Installation and Dependences

```bash
git clone https://github.com/TheJacksonLab/gECG_thiophene.git 
```

To run this project, you will need to set up a Python environment. We recommend using Conda, but others should work too.

```bash
conda create --name your_env_name python=3.11
```

Activate the new environment

```bash
conda activate your_env_name
```

Install the required packages

```bash
pip install -r requirements.txt
```

```bash
cd gECG_thiophene/
pip install -e .
```

## Usage

### Generate new polymer data

./data_generation/ provides the baisc codes for generating the polymer datasets. The workflow are as follows:

1. Creating polymer sequences based on SMILES using generate_polymer.py
    - It utlizes rdkit
    - Change length in lengths_list to control the degree of polymerization

2. (Optional) Sampling conformations with molecular dynamics simulations
    - Prepare the inputs for Lammps with the OPLS-AA force fields inputs
        - Can be conducted with LigPargen server
        - Or locally with generate_lmp.py, which needs BOSS installed locally (refer to [Local ligPargen](https://www.linkedin.com/pulse/how-install-ligpargen-server-locally-leela-sriram-dodda/))
    - Run MD simulations
        - Sample inputs in ./data_generation/tp_gen/in.nvt*
    - Collect confomrations from MD trajectories and prepare them in the format of ORCA inputs
        - Refer to ./data_generation/run_MD.py

3. Performing QM calculations with ORCA
    - Sample inputs for ZINDO/S and DFT are provided in ./data_generation/tp_gen/*.inp

4. Collect ORCA inputs and prepare datasets
    1. AA resolution: ./data_generation/collect_data*.py
    2. CG resolutions: ./data_generation/CG/CG*.py

### Train gECG models

Once the dataset is generated or downloaded (see the Data section below), the gECG model can be trained.

For a quick test of the model training and evalution,

```bash
cd scripts/
python train_CG3R.py
```

For complete training the gECG models at various CG resolutions, change the datasets accordingly.

### gECG model inference

For a sample model inference and evaluation,

```bash
cd scripts/
python evaluate.py
```

The outputs are saved into the `output.txt'. The MAE and R2 are printed.

### Fine-tuning of gECG models

One example of fine-tuning gECG models is provided. It shows how to improve the model from the ZINDO/S precision to the DFT precision with a small DFT dataset.

```bash
cd scripts/
python fine_tuning.py
```

It involves a two-step (partial-then-full) fine-tuning, which in total takes around 5-10 min on laptop CPU.

## Data

Because the datasets are too large, they need to be downloaded from [Zenodo]

### Processed data

Datasets of CG polymers at various resolutions are included. The details of the CG representations are provided in the paper.

For loading data,

```python
import torch
dataset = torch.load(dataset_to_load)
```

### Raw data

Output files directly from quantum mechanical calculations implemented with ORCA
