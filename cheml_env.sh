#!/bin/bash

# create & activate conda environment
conda create -n cheml python=3.9 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cheml

# rdkit
pip install rdkit-pypi 

# torch
## cpu version
# pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu 
# pip install torch_geometric 
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html 

## gpu version (cuda 11.7)
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 
# pip install torch_geometric 
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html 

# ML packages
pip install scikit-learn==1.3.0
# pip install xgboost==1.7.6
pip install xgboost==1.3.1
pip install lightgbm==3.3.5 

# other packages
pip install pandas matplotlib seaborn tqdm
pip install dash==2.10.2 
pip install molplotly 
pip install ipykernel 