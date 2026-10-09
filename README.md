# SAMD-YOLO

**Surface-Semantic-Aware Industrial Defect Detection**

This repository provides the research implementation of SAMD-YOLO, a surface-semantic-aware defect detection framework for industrial component inspection.

## Main Components

* **Surface-Semantic Guidance:** An auxiliary segmentation branch modeling background, cast, and machined surfaces.
* **Deformable Cross-Task Attention (DCA):** Incorporates surface-semantic features into multi-scale defect detection.
* **Teacher–Student Distillation:** Transfers surface-semantic predictions and feature information from a segmentation teacher to the detection student.
* **Low-Frequency Spatial Cosine Distillation:** Encourages spatial alignment of low-frequency semantic features.

## Code Structure

* `MTL_M5_6/sam_net/ultralytics/`: Modified Ultralytics detection framework.
* `MTL_M5_6/scripts/`: Training, validation, and teacher-related scripts.

## Data and Model Weights

Large datasets and model checkpoints are not included in this source-code repository. Dataset access and preprocessing requirements should be followed according to the associated manuscript.

## Reproducibility

The implementation requires a compatible Python and PyTorch environment. Dataset paths and pretrained teacher checkpoints must be configured before training.

## Acknowledgments

This implementation builds upon the Ultralytics framework. The original framework's applicable license and copyright notices must be preserved.
