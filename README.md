# heart-seg

Cardiac structure segmentation from CT images.  
This repository is organized into multiple branches — see below for guidance.

## 🌿 Branch Structure

| Branch                        | Description                                                                                                                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ct_base_paper`               | **Main development branch.** Contains the current codebase used for the paper experiments, including model training pipelines (SegResNet, UNet++, SwinUNETR) and conformal prediction integration. |
| `hfs_legacy`                  | Legacy code from the HFS phase of the project. Kept for reference and reproducibility of earlier results.                                                                                          |
| `ct-dataset`                  | Dataset preparation and preprocessing scripts for CT data.                                                                                                                                         |
| `feat-all-hyperparameter-cfg` | Feature branch for unified hyperparameter configuration.                                                                                                                                           |
| `improve_visualization`       | Experimental branch for visualization improvements.                                                                                                                                                |

## 🚀 Getting Started

Clone the repository and switch to the active development branch:

```bash
git clone https://github.com/ligia-ufpe/heart-seg.git
cd heart-seg
git checkout ct_base_paper
```
