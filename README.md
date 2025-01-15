# Crack-oriented contrastive learning for enhanced surface defect detection: a universally applicable approach

***The code will be released after the publication of the corresponding paper.***

## Overview
![COCL](images/COCL.png)

## Preparation
### Environments
```
Python: 3.9.19
PyTorch: 2.3.1
Torchvision: 0.18.1
CUDA: 11.8
NumPy: 1.26.3
PIL: 10.2.0
timm: 1.0.9
```

### Dataset
**Semiconductor die dataset cannot be disclosed due to considerations of security and confidentiality.**

MixedWM38 : [WaferMap Dataset: MixedWM38](https://github.com/Junliangwangdhu/WaferMap?tab=readme-ov-file)

The original shape of MixedWM38 is incompatible with pre-trained models, necessitating minor preprocessing. The raw data was reshaped to (224, 224, 3) and underwent label encoding. The processed dataset is available in the directory ```COCL/datasets/MixedWM38/```. The processed dataset has been compressed into a tar.gz file, which can be extracted using the following code:
```
tar -xzvf {filename}.tar.gz
```

## Usage
### Data augmentation
TBD
```
python preprocess_mixedwm38.py
```
