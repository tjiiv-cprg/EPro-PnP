# EPro-PnP

📢 **NEWS:** We have released [EPro-PnP-v2](https://github.com/tjiiv-cprg/EPro-PnP-v2). A new updated preprint can be found on [arXiv](https://arxiv.org/abs/2303.12787).

**EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation**
<br>
In CVPR 2022 (Oral, **Best Student Paper**). [[paper](https://arxiv.org/pdf/2203.13254.pdf)][[video](https://www.youtube.com/watch?v=TonBodQ6EUU)]
<br>
[Hansheng Chen](https://lakonik.github.io/)\*<sup>1,2</sup>, [Pichao Wang](https://wangpichao.github.io/)†<sup>2</sup>, [Fan Wang](https://scholar.google.com/citations?user=WCRGTHsAAAAJ&hl=en)<sup>2</sup>, [Wei Tian](https://scholar.google.com/citations?user=aYKQn88AAAAJ&hl=en)†<sup>1</sup>, [Lu Xiong](https://www.researchgate.net/scientific-contributions/Lu-Xiong-71708073)<sup>1</sup>, [Hao Li](https://scholar.google.com/citations?user=pHN-QIwAAAAJ&hl=zh-CN)<sup>2</sup>

<sup>1</sup>Tongji University, <sup>2</sup>Alibaba Group
<br>
\*Part of work done during an internship at Alibaba Group.
<br>
†Corresponding Authors: Pichao Wang, Wei Tian.

## Introduction

EPro-PnP is a probabilistic Perspective-n-Points (PnP) layer for end-to-end 6DoF pose estimation networks. Broadly speaking, it is essentially a continuous counterpart of the widely used categorical Softmax layer, and is theoretically generalizable to other learning models with nested <!-- $\mathrm{arg\,min}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5Cmathrm%7Barg%5C%2Cmin%7D"> optimization.

<img src="intro.png" width="500"  alt=""/>

Given the layer input: an <!-- $N$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?N">-point correspondence set <!-- $X = \left\{x^\text{3D}_i,x^\text{2D}_i,w^\text{2D}_i\,\middle|\,i=1\cdots N\right\}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?X%20%3D%20%5Cleft%5C%7Bx%5E%5Ctext%7B3D%7D_i%2Cx%5E%5Ctext%7B2D%7D_i%2Cw%5E%5Ctext%7B2D%7D_i%5C%2C%5Cmiddle%7C%5C%2Ci%3D1%5Ccdots%20N%5Cright%5C%7D"> consisting of 3D object coordinates <!-- $x^\text{3D}_i \in \mathbb{R}^3$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?x%5E%5Ctext%7B3D%7D_i%20%5Cin%20%5Cmathbb%7BR%7D%5E3">, 2D image coordinates <!-- $x^\text{2D}_i \in \mathbb{R}^2$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?x%5E%5Ctext%7B2D%7D_i%20%5Cin%20%5Cmathbb%7BR%7D%5E2">, and 2D weights <!-- $w^\text{2D}_i \in \mathbb{R}^2_+ $ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?w%5E%5Ctext%7B2D%7D_i%20%5Cin%20%5Cmathbb%7BR%7D%5E2_%2B">, a conventional PnP solver searches for an optimal pose <!-- $y^\ast$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?y%5E%5Cast"> (rigid transformation in SE(3)) that minimizes the weighted reprojection error. Previous work tries to backpropagate through the PnP operation, yet <!-- $y^\ast$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?y%5E%5Cast"> is inherently non-differentiable due to the inner <!-- $\mathrm{arg\,min}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?%5Cmathrm%7Barg%5C%2Cmin%7D"> operation. This leads to convergence issue if all the components in <!-- $X$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?X"> must be learned by the network.

In contrast, our probabilistic PnP layer outputs a posterior distribution of pose, whose probability density <!-- $p(y|X)$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?p(y%7CX)"> can be derived for proper backpropagation. The distribution is approximated via Monte Carlo sampling. With EPro-PnP, the correspondences <!-- $X$ --> <img style="transform: translateY(0.1em); background: white;" src="https://latex.codecogs.com/svg.latex?X"> can be learned from scratch altogether by minimizing the KL divergence between the predicted and target
pose distribution.

## Models

### V1 models in this repository

#### **[EPro-PnP-6DoF](EPro-PnP-6DoF) for 6DoF pose estimation**<br>
  <img src="EPro-PnP-6DoF/viz.gif" width="500" alt=""/>

#### **[EPro-PnP-Det](EPro-PnP-Det) for 3D object detection**

  <img src="EPro-PnP-Det/resources/viz.gif" width="500" alt=""/>

### New V2 models

#### **[EPro-PnP-Det v2](https://github.com/tjiiv-cprg/EPro-PnP-v2/tree/main/EPro-PnP-Det_v2): state-of-the-art monocular 3D object detector**

Main differences to [v1b](EPro-PnP-Det):

- Use GaussianMixtureNLLLoss as auxiliary coordinate regression loss
- Add auxiliary depth and bbox losses
  
At the time of submission (Aug 30, 2022), EPro-PnP-Det v2 **ranks 1st** among all camera-based single-frame object detection models on the [official nuScenes benchmark](https://www.nuscenes.org/object-detection?externalData=no&mapData=no&modalities=Camera) (test split, without extra data).

| Method                                                   | TTA | Backbone |    NDS    |    mAP    |   mATE    |   mASE    |   mAOE    |   mAVE    |   mAAE    | Schedule |
|:---------------------------------------------------------|:---:|:---------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:--------:|
| EPro-PnP-Det v2 (ours)                                   |  Y  | R101     | **0.490** |   0.423   |   0.547   | **0.236** | **0.302** |   1.071   |   0.123   |  12 ep   |
| [PETR](https://github.com/megvii-research/petr)          |  N  | Swin-B   |   0.483   | **0.445** |   0.627   |   0.249   |   0.449   |   0.927   |   0.141   |  24 ep   |
| [BEVDet-Base](https://github.com/HuangJunJie2017/BEVDet) |  Y  | Swin-B   |   0.482   |   0.422   | **0.529** | **0.236** |   0.395   |   0.979   |   0.152   |  20 ep   |
| EPro-PnP-Det v2 (ours)                                   |  N  | R101     |   0.481   |   0.409   |   0.559   |   0.239   |   0.325   |   1.090   | **0.115** |  12 ep   |
| [PolarFormer](https://github.com/fudan-zvg/PolarFormer)  |  N  | R101     |   0.470   |   0.415   |   0.657   |   0.263   |   0.405   | **0.911** |   0.139   |  24 ep   |
| [BEVFormer-S](https://github.com/zhiqi-li/BEVFormer)     |  N  | R101     |   0.462   |   0.409   |   0.650   |   0.261   |   0.439   |   0.925   |   0.147   |  24 ep   |
| [PETR](https://github.com/megvii-research/petr)          |  N  | R101     |   0.455   |   0.391   |   0.647   |   0.251   |   0.433   |   0.933   |   0.143   |  24 ep   |
| [EPro-PnP-Det v1](EPro-PnP-Det_v2)                       |  Y  | R101     |   0.453   |   0.373   |   0.605   |   0.243   |   0.359   |   1.067   |   0.124   |  12 ep   | 
| [PGD](https://github.com/open-mmlab/mmdetection3d)       |  Y  | R101     |   0.448   |   0.386   |   0.626   |   0.245   |   0.451   |   1.509   |   0.127   | 24+24 ep |
| [FCOS3D](https://github.com/open-mmlab/mmdetection3d)    |  Y  | R101     |   0.428   |   0.358   |   0.690   |   0.249   |   0.452   |   1.434   |   0.124   |    -     |

#### **[EPro-PnP-6DoF v2](https://github.com/tjiiv-cprg/EPro-PnP-v2/tree/main/EPro-PnP-6DoF_v2) for 6DoF pose estimation**<br>

Main differences to [v1b](EPro-PnP-6DoF):

- Fix w2d scale handling **(very important)**
- Improve network initialization
- Adjust loss weights

With these updates the v2 model can be trained **without 3D models** to achieve better performance (ADD 0.1d = 93.83) than [GDRNet](https://github.com/THU-DA-6D-Pose-Group/GDR-Net) (ADD 0.1d = 93.6), unleashing the full potential of simple end-to-end training.

## Use EPro-PnP in Your Own Model

We provide a [demo](demo/fit_identity.ipynb) on the usage of the EPro-PnP layer.

## Citation

If you find this project useful in your research, please consider citing:

```
@inproceedings{epropnp, 
  author = {Hansheng Chen and Pichao Wang and Fan Wang and Wei Tian and Lu Xiong and Hao Li, 
  title = {EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation}, 
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  year = {2022}
}
```
