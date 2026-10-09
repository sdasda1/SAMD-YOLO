# SAMD-YOLO

**Surface-Semantic-Guided Industrial Defect Detection**

This repository contains the research implementation of **SAMD-YOLO**, an industrial defect detection framework that uses surface-semantic information to guide multi-scale detection through auxiliary segmentation, deformable cross-task attention, and teacher–student distillation.

## 1. Overview

Industrial components can contain visually different surface regions, including **cast** and **machined** surfaces. SAMD-YOLO incorporates these surface semantics into the detection process rather than treating all regions as having identical visual characteristics.

The framework is built on a modified Ultralytics YOLO implementation and uses a segmentation teacher during training.

## 2. Main Components

- **Surface-Semantic Segmentation:** An auxiliary branch predicts three categories: **Background**, **Cast**, and **Machined**.
- **Deformable Cross-Task Attention (DCA):** Surface-semantic features guide the multi-scale detection features using adaptive spatial sampling and fusion.
- **Teacher–Student Distillation:** A ResNeSt50-Attention-UNet teacher provides soft segmentation targets and intermediate features during student training.
- **Low-Frequency Spatial Cosine Distillation:** Teacher and student features are spatially smoothed and compared using cosine similarity over spatial positions. This is a spatial low-pass operation, not an explicit Fourier transform.

## 3. Repository Structure

- `SAMD-YOLO/sam_net/ultralytics/` — Modified Ultralytics framework and model implementation.
- `SAMD-YOLO/scripts/final_train.py` — Final training entry point.
- `SAMD-YOLO/scripts/train_mtl.py` — Multi-task training script.
- `SAMD-YOLO/scripts/val_final.py` — Validation script.
- `SAMD-YOLO/scripts/models/train_ResNeSt50_UNet_final.py` — Segmentation teacher training script.

## 4. Core Implementation Files

The primary framework modifications are located in:

- `SAMD-YOLO/sam_net/ultralytics/nn/modules/head.py` — Surface-aware detection head and DCA.
- `SAMD-YOLO/sam_net/ultralytics/nn/modules/__init__.py` — Custom module registration.
- `SAMD-YOLO/sam_net/ultralytics/nn/tasks.py` — Model construction and parsing.
- `SAMD-YOLO/sam_net/ultralytics/engine/trainer.py` — Teacher integration during training.
- `SAMD-YOLO/sam_net/ultralytics/utils/loss.py` — Detection, segmentation, and distillation losses.

## 5. Data, Weights, and Reproduction

The training workflow requires industrial defect annotations, surface-semantic supervision for the teacher, and the corresponding teacher checkpoint. Large datasets and pretrained model weights are **not included** in this repository.

Before running training or validation, users must configure dataset paths, model configuration files, pretrained checkpoints, and software dependencies for their environment. Some scripts may contain machine-specific paths that require adjustment. The source-code release alone should not be interpreted as a fully automated, one-command reproduction package.

The accompanying manuscript describes the method and experimental setup, including external evaluation involving the HSS-IAD Casting dataset.

## 6. Acknowledgments and Licensing

This project builds upon the **Ultralytics** framework and **PyTorch**. Original third-party copyright notices and applicable license terms must be retained and respected. Refer to the license files included with the underlying framework.

## 7. Code Availability

Repository: https://github.com/sdasda1/SAMD-YOLO