# Adversarial Robustness of ViM vs ViT vs ResNet

This repository contains my results for the DD2424 project at KTH.
In short, this project aimed to investigate the baseline adversarial robustness of the 
new Vision Mamba architecture published by [HUST researchers](https://github.com/hustvl/Vim).

For further details, see the brief report in ```ProjectReport.pdf```.

## Overview

Overview of the relevant parts of the repository structure and file contents
    
    ```bash
    .
    ├── images/                             # Visuals generated for the report
    ├── mamba_ssm/                          # Modified mamba_ssm from https://github.com/hustvl/Vim
    ├── vim/                                # ViM model implementation from https://github.com/hustvl/Vim
    ├── datasetProcessor.ipynb              # Notebook for dataset processing and preparation
    ├── fgsm.ipynb                          # Fast Gradient Sign Method implementation
    ├── genAdvDataset.py                    # Script to generate adversarial datasets
    ├── landscapeVisuals.ipynb              # Notebook for generating landscape visualizations
    ├── loadVim.py                          # Utility functions to load the ViM model
    ├── ProjectReport.pdf                   # Final project report document
    ├── robustnessAnalysisBaseline.ipynb    # Notebook for baseline robustness analysis
    ├── trainingVisuals.ipynb               # Notebook for visualizing training metrics
    ├── transferAttack.ipynb                # Notebook for transfer attack experiments
    ├── vimGenAdv.ipynb                     # Notebook to generate adversarial examples for ViM model
    ├── vimTrainLoop.ipynb                  # Notebook with ViM model training implementation
    ```


## Setup

To use the notebooks, you need to set up a virtual environment as instructed in setup. 
Additionally, you need the imagenette dataset in the root folder. (Or change the paths in the code.)

```bash
conda create -p .conda python=3.12 -y
```

```bash
conda init
conda activate ./.conda
python -m pip install --upgrade pip
conda install cudatoolkit==11.8 -c nvidia -y
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc -y
conda install packaging -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Generate Adversarial Dataset

### Baseline

Below are example scripts to generate an adversarial dataset for the validation data of imagenette with microsoft/resnet-152 in ./data_adv_resnet and google/vit-base-patch16-224 in ./data_adv_vit respectively. Generation for Vim models take place in the notebook ```vimGenAdv.ipynb```

```bash
python ./genAdvDataset.py --path "./data_adv_resnet" --dataset "./imagenette/imagenette2/val" --model "microsoft/resnet-152"
```

```bash
python ./genAdvDataset.py --path "./data_adv_vit" --dataset "./imagenette/imagenette2/val" --model "google/vit-base-patch16-224"
```

### Celeb Data

```bash
python ./genAdvDataset.py --path "./CelebAdv/CelAdvResNetA" --dataset "./CelebSubset/CelebVal" --model "microsoft/resnet-152" --checkpoint "./models/ResNetA.pth"
```

```bash
python ./genAdvDataset.py --path "./CelebAdv/CelAdvResNetB" --dataset "./CelebSubset/CelebVal" --model "microsoft/resnet-152" --checkpoint "./models/ResNetB.pth"
```

```bash
python ./genAdvDataset.py --path "./CelebAdv/CelAdvVitA" --dataset "./CelebSubset/CelebVal" --model "google/vit-base-patch16-224" --checkpoint "./models/VitA.pth"
```

```bash
python ./genAdvDataset.py --path "./CelebAdv/CelAdvVitB" --dataset "./CelebSubset/CelebVal" --model "google/vit-base-patch16-224" --checkpoint "./models/VitB.pth"
```

### Robustness of Adversarially Trained Models

```bash
python ./genAdvDataset.py --path "./AdvRobustnessData/RobustnessResNetA" --dataset "./CelebSubset/CelebTest" --model "microsoft/resnet-152" --checkpoint "./models/AdvResNetA.pth"
```

```bash
python ./genAdvDataset.py --path "./AdvRobustnessData/RobustnessVitA" --dataset "./CelebSubset/CelebTest" --model "google/vit-base-patch16-224" --checkpoint "./models/AdvVitA.pth"
```

### Robustness with blurred inputs

```bash
python ./genAdvDataset.py --blur --path "./BlurRobustnessData/BlurResNetA" --dataset "./CelebSubset/CelebTest" --model "microsoft/resnet-152" --checkpoint "./models/ResNetA.pth"
```

```bash
python ./genAdvDataset.py --blur --path "./BlurRobustnessData/BlurVitA" --dataset "./CelebSubset/CelebTest" --model "google/vit-base-patch16-224" --checkpoint "./models/VitA.pth"
```

## Setup Vim - Not working, only linux

First create a Conda virtual environment:

```bash
conda create -p .conda python=3.10.13 -y
```

Then install the requirements into the environment

```bash
conda init
conda activate ./.conda
python -m pip install --upgrade pip
conda install cudatoolkit==11.8 -c nvidia -y
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc -y
conda install packaging
```

Make sure to have Microsoft C++ Build TOols >=14.0 installed

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r ./vim/vim/vim_requirements.txt
pip install -r requirements.txt
```

```bash
pip install -e ./vim/causal-conv1d
pip install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl
pip install -e ./vim/mamba-1p1p1
```

