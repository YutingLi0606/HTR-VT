# (Pattern Recognition)HTR-VT
Pytorch implementation of paper "HTR-VT: Handwritten Text Recognition with Vision Tranformer"

[[Project Page]](https://yutingli0606.github.io/HTR-VT/)
[[ArXiv]](https://www.sciencedirect.com/science/article/abs/pii/S0031320324007180) 
[[Google Drive]]()

If our project is helpful for your research, please consider citing :
```
@article{LI2025110967,
title = {HTR-VT: Handwritten text recognition with vision transformer},
journal = {Pattern Recognition},
volume = {158},
pages = {110967},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.110967},
author = {Yuting Li and Dexiong Chen and Tinglong Tang and Xi Shen},
}
```

## Table of Content
* [1. Overview](#1-overview)
* [2. Visual Results](#2-visual-results)
* [3. Installation](#3-installation)
* [4. Quick Start](#4-quick-start)
* [5. Acknowledgement](#6-acknowledgement)

## 1. Overview
<p align="center">
<img src="img/HTR-VT.png" width="500px" alt="teaser">
</p>

## 2. Visual Results
<p align="center">
<img src="img/visual.png" width="1000px" alt="method">
</p>

## 3. Installation

### 3.1. Environment

Our model can be learnt in a **single GPU RTX-4090 24G**
```bash
conda env create -f environment.yml
conda activate htr
```

The code was tested on Python 3.9 and PyTorch 1.13.0.


### 3.2. Datasets

* Using **IAM, READ2016 and LAM** for handwritten text recognition.
* Download datasets to ./data/ and split into train/val/test.
Take IAM for an example:

The structure of the file should be:
```
./data/iam/
├── train.ln
├── val.ln
├── test.ln
└── lines
```
* We have already split Tiny-imagenet, you can download it from [here.](https://drive.google.com/drive/folders/1xT-cX22_I8h5yAYT1WNJmhSLrQFZZ5t1?usp=sharing)


## 4. Quick Start
* Our model checkpoints are saved [here.](https://drive.google.com/drive/folders/1xT-cX22_I8h5yAYT1WNJmhSLrQFZZ5t1?usp=sharing)
* 
* We provide convenient and comprehensive commands in ./run/ to train and test on different datasets to help researchers reproducing the results of the paper.


## 6. Acknowledgement

We appreciate helps from public code like [VAN](https://github.com/FactoDeepLearning/VerticalAttentionOCR) and [OrigamiNet](https://github.com/IntuitionMachines/OrigamiNet).  
