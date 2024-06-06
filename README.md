# gECG_thiophene

Code for Generalized ECG (gECG) for thiophene polymers paper

## Introduction

Electronic Coarse Graining (ECG) is a machine learning approach capable of predicting quantum mechanical (QM) optoelectronic properties of molecules (and materials) based directly on molecular configurations at the coarse-grained (CG) resolution.

This project involves the development of generalized ECG (gECG) models for thiophene polymers, including the necessary codes, datasets, and trained models. More information can be found in our preprint paper, "Generalized Electronic Coarse Graining for Thiophene Polymers," available [here](https://!!!).

## Installation and Dependencies

Clone the repository and set up the required environment:

```bash
git clone https://github.com/TheJacksonLab/gECG_thiophene.git
cd gECG_thiophene/

# Create a Conda environment
conda create --name your_env_name python=3.11
conda activate your_env_name

# Install required Python packages
pip install -r requirements.txt
pip install -e .
```

## Usage

### Generate new polymer data

The `data_generation/` directory provides the basic codes for generating polymer datasets. The workflow is as follows:

1. **Create Polymer Sequences**:
   - Use `generate_polymer.py` to create polymer sequences using `rdkit`.
   - Modify `lengths_list` in the script to control the degree of polymerization.

2. **Optional: Sample Conformations with Molecular Dynamics**:
   - Prepare the inputs for Lammps with the OPLS-AA force fields.
     - This can be conducted with the LigPargen server or locally with `generate_lmp.py`, which requires [BOSS installed locally](https://www.linkedin.com/pulse/how-install-ligpargen-server-locally-leela-sriram-dodda/).
   - Run MD simulations.
     - Sample inputs are provided in `./data_generation/tp_gen/in.nvt*`.
   - Collect conformations from MD trajectories and prepare them in the format of ORCA inputs.
     - Refer to `./data_generation/run_MD.py`.

3. **Perform QM Calculations with ORCA**:
   - Sample inputs for ZINDO/S and DFT are provided in `./data_generation/tp_gen/*.inp`.

4. **Collect ORCA Inputs and Prepare Datasets**:
   - AA resolution: `./data_generation/collect_data*.py`
   - CG resolutions: `./data_generation/CG/CG*.py`

### Train gECG models

Once the dataset is generated or downloaded (see the Data section below), the gECG model can be trained. The model architecture is based on [ComENet](https://arxiv.org/abs/2206.08515).

For a quick test of the model training and evalution,

```bash
cd scripts/
python train_CG3R.py
```

For completely training the gECG models at various CG resolutions, change the datasets accordingly.

### gECG model inference

For a sample model inference and evaluation,

```bash
cd scripts/
python evaluate.py
```

The outputs are saved into the `output.txt`. The MAE and R2 are printed.

### Fine-tuning of gECG models

One example of fine-tuning gECG models is provided. It shows how to improve the model from the ZINDO/S precision to the DFT precision with a small DFT dataset.

```bash
cd scripts/
python fine_tuning.py
```

It involves a two-step (partial-then-full) fine-tuning, which in total takes around 5-10 min on a laptop CPU.

## Data

Due to the large size of the datasets, they need to be downloaded from [Zenodo](!!!)

### Processed data

Datasets of CG polymers at various resolutions are included. Details of the CG representations are provided in the paper.

For loading data,

```python
import torch
dataset = torch.load(dataset_to_load)
```

### Raw data

Output files directly from QM calculations implemented with ORCA
