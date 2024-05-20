# Adversarial Robustness of ViT vs ResNet

This is our group project for DD2424 at KTH.

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

Below are example scripts to generate an adversarial dataset for the validation data of imagenette with microsoft/resnet-152 in ./data_adv_resnet and google/vit-base-patch16-224 in ./data_adv_vit respectively.

```bash
python ./genAdvDataset.py --path "./data_adv_resnet" --dataset "./imagenette/imagenette2/val" --model "microsoft/resnet-152"
```

```bash
python ./genAdvDataset.py --path "./data_adv_vit" --dataset "./imagenette/imagenette2/val" --model "google/vit-base-patch16-224"
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

