# Results

This directory contains a summary of the experimental results reported for the UNETR2D-MSTF framework.

The complete experimental outputs, trained model weights, prediction masks, and dataset files are not included in this public repository.

## Main Results

The framework was evaluated on two medical image segmentation benchmarks:

- **REFUGE2** — optic disc and optic cup segmentation
- **ISIC 2018** — skin lesion segmentation

### Overall Performance

| Configuration | REFUGE2 Disc Dice | REFUGE2 Cup Dice | ISIC 2018 Dice |
|---|---:|---:|---:|
| UNETR2D-MSTF | 94.8% | 86.4% | 88.2% |
| + 5-Fold CV | 94.3% | 86.3% | **89.7%** |
| + TTA + STAPLE | **95.2%** | **86.7%** | 88.7% |

The reported results show that 5-fold cross-validation improves the ISIC 2018 Dice score, while TTA with STAPLE produces the highest reported optic disc Dice on REFUGE2. :contentReference[oaicite:0]{index=0}

## Ablation Study

The contribution of the individual MSTF branches was evaluated using five configurations.

| Configuration | REFUGE2 Mean Dice | Disc Dice | Cup Dice | ISIC 2018 Mean Dice |
|---|---:|---:|---:|---:|
| Baseline (No MSTF) | 85.43 | 86.37 | 84.50 | 85.40 |
| MSTF (Local Only) | 86.53 | 87.36 | 85.69 | 86.50 |
| MSTF (Global Only) | 85.12 | 86.31 | 83.92 | 86.70 |
| MSTF (Dilated Only) | 85.83 | 86.80 | 84.86 | 86.10 |
| **Full MSTF (Ours)** | **87.20** | **88.20** | **86.10** | **87.50** |

The full MSTF configuration achieved the highest REFUGE2 mean, optic disc, and optic cup Dice scores among the ablation configurations. :contentReference[oaicite:1]{index=1}

## Boundary Metrics

Additional boundary-based evaluation was performed using:

- **Average Surface Distance (ASD)**
- **95th Percentile Hausdorff Distance (95HD)**

For REFUGE2, the reported ASD values are **0.4125 pixels for the optic disc** and **1.3218 pixels for the optic cup**. The corresponding 95HD values are **2.3588 pixels** and **3.2804 pixels**, respectively. :contentReference[oaicite:2]{index=2}

For ISIC 2018, the reported ASD is **6.1885 pixels** and the 95HD is **14.1314 pixels**. :contentReference[oaicite:3]{index=3}

## Notes

- Results shown here are from the submitted research work.
- The datasets are not distributed with this repository.
- Trained model weights and complete experimental outputs are not publicly released.
- Reproduction may require obtaining the original datasets and configuring the corresponding training environment.
