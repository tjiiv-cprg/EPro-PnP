## Prerequisites

The code has been tested in the environment described as follows:

- Linux (tested on Ubuntu 18.04/20.04 LTS)
- Python 3.7
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 11
- [PyTorch](https://pytorch.org/get-started/previous-versions/) 1.10.1
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) 0.6.1
- [MMCV](https://github.com/open-mmlab/mmcv) 1.4.1

An example script for installing the python dependencies under CUDA 11.3:

```bash
# Export the PATH of CUDA toolkit
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

# Create conda environment
conda create -y -n epropnp_det python=3.7
conda activate epropnp_det
conda install -y pip

# Install pytorch
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV
pip install mmcv-full==1.4.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install Pytorch3D dependencies
conda install -y -c fvcore -c iopath -c conda-forge -c bottler fvcore iopath nvidiacub

# Install Pytorch3D from source
git clone https://github.com/facebookresearch/pytorch3d
cd pytorch3d && git checkout v0.6.1 && pip install -v -e . && cd ..
# alternatively if you use pytorch 1.10.0, PyTorch3D can be directly installed via conda:
# conda install -y pytorch3d==0.6.1 -c pytorch3d
```

## Installation

Clone the repository and install epropnp_det:

```bash
git clone https://github.com/tjiiv-cprg/EPro-PnP && cd EPro-PnP/EPro-PnP-det
pip install -v -e .
```

## Verification

To verify the installation, you can download one of the checkpoint files [[Google Drive](https://drive.google.com/drive/folders/1AWRg09fkt66I8rgrp33Lwb9l6-D6Gjrg) | [Baidu Pan](https://pan.baidu.com/s/1j7xgkwD-rcxHMaNupRP_bQ?pwd=cx5b#list/path=%2FEPro-PnP-Det)] and run the inference demo:

```bash
python demo/infer_imgs.py demo/ /PATH/TO/CONFIG /PATH/TO/CHECKPOINT --show-views 3d bev mc
```

The resulting visualizations will be saved into `demo/viz`.
