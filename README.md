# UNETR2D-MSTF: Transformer-based Medical Image Segmentation

A transformer-based medical image segmentation framework that extends the UNETR architecture with a **Multi-Scale Token Fusion (MSTF)** module for improved multi-scale feature representation.

## Overview

Medical image segmentation involves identifying and delineating anatomical or pathological structures at the pixel level. Conventional convolutional architectures are effective at learning local features but can struggle to capture long-range dependencies, while standard transformer architectures provide global context but may lack explicit multi-scale spatial representations.

**UNETR2D-MSTF** addresses this by integrating a Multi-Scale Token Fusion module into a 2D UNETR-style transformer encoder. MSTF processes spatial features through three complementary branches:

- **Local branch** – captures fine-grained local spatial information using 3×3 convolution.
- **Global branch** – captures coarse contextual information using pooling, convolution and upsampling.
- **Dilated branch** – expands the receptive field using dilated convolution.

The resulting multi-scale representation is fused and incorporated through a residual connection.

## Key Features

- 2D adaptation of the UNETR transformer-based segmentation architecture
- Multi-Scale Token Fusion (MSTF)
- Local, global and dilated spatial feature extraction
- U-Net-style decoder with transformer skip connections
- Deep supervision during training
- 5-fold cross-validation for model generalization
- Test-Time Augmentation (TTA)
- STAPLE-based probabilistic label fusion
- Evaluation using Dice, IoU, ASD and 95HD

## Architecture

The proposed pipeline consists of four major stages:

1. **Patch-based Tokenization**
2. **Transformer Encoding with MSTF**
3. **CNN-based Decoder with Deep Supervision**
4. **Robust Inference using 5-Fold Cross-Validation, TTA and STAPLE**

Input images are resized to **256 × 256** and divided into non-overlapping **16 × 16 patches**. Each patch is projected into a **512-dimensional embedding space** before being processed by the transformer encoder.

The transformer encoder consists of **12 layers**, with MSTF integrated into the transformer blocks. Skip connections are extracted from transformer depths 3, 6, 9 and 12 and passed to the decoder.

## MSTF Module

The Multi-Scale Token Fusion module operates on the spatial representation of transformer tokens through three parallel branches:

```text
                    Transformer Features
                            |
             +--------------+--------------+
             |              |              |
          Local           Global         Dilated
          3×3 Conv      Pool + Conv       3×3 Conv
             |              |              |
             +--------------+--------------+
                            |
                     Feature Fusion
                            |
                    1×1 Projection
                            |
                    Residual Addition
```
This allows the model to combine local boundary information, global contextual information, and expanded receptive-field features while retaining the global modeling capability of the transformer.

## Datasets

The framework was evaluated on two medical image segmentation benchmarks:

### REFUGE2

Used for retinal fundus image segmentation, with separate segmentation of:

- Optic Disc
- Optic Cup

### ISIC 2018

Used for binary skin lesion segmentation.

> **Note:** The datasets are not included in this repository. Please obtain them from their respective official sources and follow their terms of use.

## Preprocessing

The experimental pipeline includes:

- Image resizing to 256 × 256
- Image normalization
- Mask processing
- Data augmentation
- Rotation and flipping
- Elastic deformation
- Zooming
- Contrast-related augmentation

## Training

The implementation is based on **TensorFlow**.

Main training components include:

- **Optimizer:** Adam
- **Loss:** 0.5 × Categorical Cross-Entropy + 0.5 × Dice Loss
- **Learning Rate:** Cosine annealing learning-rate schedule
- **Warm-up:** Linear warm-up
- **Deep Supervision**
- **Data Augmentation**

## Evaluation

The framework is evaluated using:

- **Dice Similarity Coefficient (Dice)**
- **Intersection over Union (IoU)**
- **Average Surface Distance (ASD)**
- **95th Percentile Hausdorff Distance (95HD)**

For REFUGE2, metrics are reported separately for optic disc and optic cup segmentation.

## Inference Enhancement

### 5-Fold Cross-Validation

Five-fold cross-validation is used to improve the reliability of model evaluation and generalization. The model is retrained for each fold, with the best checkpoint selected according to validation Dice performance.

### Test-Time Augmentation + STAPLE

During inference, multiple transformed versions of an input image are generated using geometric and photometric transformations. The resulting predictions are combined using **STAPLE (Simultaneous Truth and Performance Level Estimation)** to obtain a probabilistic fused segmentation.

## Ablation Study

The contribution of the MSTF branches is investigated through the following configurations:

- Baseline UNETR without MSTF
- MSTF with Local branch only
- MSTF with Global branch only
- MSTF with Dilated branch only
- Full MSTF

The ablation study evaluates the contribution of the individual branches and their combined multi-scale representation.

## Selected Results

The submitted study reports the following representative results:

| Dataset | Configuration | Dice |
|---|---|---:|
| REFUGE2 – Disc | Baseline UNETR2D-MSTF | 94.8% |
| REFUGE2 – Disc | + TTA + STAPLE | **95.2%** |
| REFUGE2 – Cup | + TTA + STAPLE | **86.7%** |
| ISIC 2018 | Baseline UNETR2D-MSTF | 88.2% |
| ISIC 2018 | + 5-Fold CV | **89.7%** |
| ISIC 2018 | + TTA + STAPLE | 88.7% |

Additional evaluation includes **IoU, ASD, and 95HD**.
## Publication

**Paper:**  
*UNETR2D-MSTF: Transformer-based Medical Image Segmentation*

**Authors:**  
Srija Mitra, Pallabi Maji, Debapriya Roy

**Conference:**  
7th International Conference on Frontiers in Computing and Systems (COMSYS 2026)

The paper was accepted as a full paper for oral presentation and was presented at COMSYS 2026. The work has been considered for publication in **Springer Lecture Notes in Networks and Systems (LNNS)**. Publication is currently pending.

## Research Status

This repository contains a **selected public implementation and project overview** of the research work.

The complete experimental codebase, unpublished research materials, datasets, trained model weights, and other research artifacts are not included in this public repository.

## Acknowledgements

This work was carried out as part of the M.Tech. Computer Science and Engineering research work at **Techno Main Salt Lake, Kolkata**.

## Citation

If you use or reference this work, please cite the published version when it becomes available.
