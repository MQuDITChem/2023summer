#!/bin/bash

# create & activate conda environment
conda create -n cheml python=3.8
conda activate cheml

# rdkit
pip install rdkit-pypi -y

# torch
## cpu version
# pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu -y
# pip install torch_geometric -y
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html -y

## gpu version (cuda 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 -y
pip install torch_geometric -y
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html -y

# ML packages
pip install scikit-learn==1.24.3 -y
pip install xgboost==1.7.6 -y
pip install lightgbm==3.3.5 -y

# other packages
# pip install pandas matplotlib seaborn tqdm
pip install dash==2.10.2 -y
pip install ipykernel -y
pip install molplotly -y