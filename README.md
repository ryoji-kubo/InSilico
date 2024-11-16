# InSilico

## Requirements
```bash
conda create --name Insilico python=3.10
# conda install -c conda-forge cxx-compiler (for HPC Only)
conda activate Insilico
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.4.0 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install ninja easydict pyyaml
# pip install numpy==1.26.4
pip install wandb
pip install pandas
conda install conda-forge::rdkit
```