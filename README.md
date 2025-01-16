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

MixedWM38 : [WaferMap Dataset](https://github.com/Junliangwangdhu/WaferMap?tab=readme-ov-file)

The original shape of MixedWM38 is incompatible with pre-trained models, necessitating minor processing. The raw data was transformed into a shape of (224, 224, 3) and subjected to label encoding. The processed dataset is available in the directory ```COCL/datasets/MixedWM38/```. The processed dataset has been compressed into a tar.gz file, which can be extracted using the following code:
```
tar -xzvf {filename}.tar.gz
```

## Usage
This is an example of running COCL on MixedWM38 dataset.

### Data augmentation
```preprocess_mixedwm38.py``` conducts canny edge detection and dilation operation.
```
python preprocess_mixedwm38.py
```

After data installation & augmentation, you may get following folder structure.
```
├── COCL
│  ├── datasets
|    ├── MixedWM38
│       └── Images
│       └── Images_CD
│       └── Labels
```

### Training Models with COCL
COCL can be executed using the following example code:

- `--network`: Specifies the backbone network. Must be one of `'resnet'`, `'densenet'`, `'efficientnet'`, `'regnet'`, or `'convnext'`.
- `--cont_loss`: Specifies the contrastive loss type. Must be one of `'cocl'`, `'supcon'`, or `'infonce'`.

```bash
root_dir=...  # Current folder path
python main.py \
  --batch_size 64 \
  --network resnet \
  --epoch 300 \
  --data ${root_dir}/datasets/MixedWM38 \
  --cont_loss cocl \
  --lambda_c 0.02 \
  --ts 0.1 \
  --num_classes 38
```

## Main results
### Semiconductor die dataset
Comparative results on semiconductor die dataset, reported as the average accuracy (%) across the five-fold cross-validation.
| Method          | ResNet50 | DenseNet121 | EfficientB4 | RegNetY032 | ConvBase | Avg. accuracy |
|-----------------|:---------------:|:------------------:|:------------------:|:-----------------:|:---------------:|:-------------:|
| *CE*(CD)       |      74.5       |        72.1        |        71.5        |       73.4        |      62.8       |      72.9     |
| *CE*(Combined) |      80.9       |        84.8        |        83.5        |       87.0        |      71.1       |      81.5     |
| *CE*(Original) |      82.2       |        84.0        |        82.4        |       83.2        |      80.0       |      82.4     |
| **COCL**       |     **86.3**    |      **86.7**      |      **84.9**      |     **87.5**      |    **86.6**     |    **86.4**   |
