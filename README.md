<p align="center">
  <h1 align="center">🧠 Autism Spectrum Disorder Detection via Facial Image Analysis</h1>
  <h3 align="center">A Comprehensive Benchmarking of Deep Learning Architectures</h3>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#manuscript">Manuscript</a> •
  <a href="#repository-structure">Structure</a> •
  <a href="#models">Models</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Journal-The%20Visual%20Computer-orange" alt="Journal: The Visual Computer">
</p>

---

> **📄 Note:** This repository contains the official implementation and experimental code accompanying the manuscript currently submitted to **The Visual Computer** (Springer). If you use this code, data, or any associated materials in your research, please cite the relevant manuscript (see [Citation](#citation)).

---

## Overview

This repository provides the complete codebase for a comprehensive benchmarking study on **Autism Spectrum Disorder (ASD) detection from facial images** using state-of-the-art deep learning architectures. The study systematically evaluates **33 deep learning models** spanning five architectural families — Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), hybrid architectures, multi-scale vision models, and face recognition networks — and proposes a novel architecture, **DHCANet (Dual-stream Hierarchical Channel Attention Network)**, for the binary classification task of distinguishing autistic from non-autistic facial images.

### Key Contributions

- **Novel proposed architecture (DHCANet)**: A dual-stream hybrid network combining DenseNet201 and MobileNetV2 backbones with custom Hierarchical Multi-Head Channel Attention (HMCAM), Temporal-Depth Convolutional Attention (TDCAM), and Inverted Residual Mobile Convolution with Attention (IRMCA) modules — achieving **93.67% test accuracy** and **0.9887 AUC-ROC**.
- **Extensive model benchmarking**: Systematic evaluation of 33 pretrained deep learning architectures across diverse model families for ASD facial image classification.
- **Rigorous data pipeline**: Multi-stage data preprocessing including deduplication, cleaning, and stratified splitting to ensure experimental reliability.
- **Comprehensive qualitative analysis**: Model interpretability via Grad-CAM heatmaps, SHAP values, and LIME explanations to identify discriminative facial regions.
- **Reproducible experimental framework**: Complete code, trained model weights, and evaluation scripts for full reproducibility of all reported results.

---

## Manuscript

> **This repository is directly associated with the following manuscript, currently under review at *The Visual Computer* (Springer):**

**Title:** *Autism Spectrum Disorder Detection via Facial Image Analysis: A Comprehensive Benchmarking of Deep Learning Architectures*

**Authors:** Ronak Panchal *et al.*

**Journal:** The Visual Computer (Springer)

**Status:** Under Review

> ⚠️ **If you use any code, data, or results from this repository, you are kindly requested to cite the above manuscript.** See the [Citation](#citation) section for the recommended BibTeX entry.

---

## Repository Structure

```
ASD-Research/
│
├── README.md                          # This file
├── autism-code-binary-detail.ipynb     # ⭐ Final proposed model (DHCANet)
│
├── part 1/                            # Phase 1: Data preparation & initial model training
│   ├── data-cleaning.ipynb            # Data cleaning and deduplication pipeline
│   ├── data-preparation.ipynb         # Dataset consolidation and splitting
│   ├── overview.ipynb                 # Exploratory data analysis & visualization
│   ├── model_metrics_extractor.ipynb  # Automated extraction of model performance metrics
│   ├── Cleaned Dataset/              # Cleaned data splits with preprocessing notebooks
│   │   ├── data-preprocessing-training-evaluation.ipynb
│   │   ├── deit-swin-convnext-mvit-clip.ipynb
│   │   ├── facenet-vggface2-insight-face.ipynb
│   │   ├── vit-alexnet-mobilenet-regnet-densenet.ipynb
│   │   └── arcface-cosface.ipynb
│   ├── tex files/                    # LaTeX tables for manuscript
│   └── model tables/                # Exported PDF performance tables
│
├── Model Training and Evaluation/    # Phase 2: Comprehensive model training
│   ├── resnet18-vgg16-19-effnet-b0-b1-b2.ipynb
│   ├── vit-alexnet-mobilenet-densenet-regnet.ipynb
│   ├── deit-swin-convnext-clip-mvit.ipynb
│   ├── densenet-201-xceptionnet.ipynb
│   ├── facenet-insightface-vggface2.ipynb
│   └── cosface-arcface.ipynb
│
├── Model results/                    # Trained model weights (.pth files)
│   └── best_<model_name>.pth        # Trained model checkpoints
│
├── Qualitative Analysis/             # Model interpretability
│   ├── GradCam/
│   │   ├── Training/                # Grad-CAM generation notebooks
│   │   └── Results/                 # Generated Grad-CAM visualizations
│   └── SHAP_Lime/
│       ├── Training/                # SHAP & LIME generation notebooks
│       └── Results/                 # Generated SHAP & LIME visualizations
│
├── Results/                          # Aggregated performance metrics
│   ├── ASD_Model_Performance_Metrics_Merged.xlsx
│   └── ASD_Model_Performance_Metrics_Simple.xlsx
│
├── AutismDataset/                    # Preprocessed dataset (train/valid/test splits)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── consolidated/
│
├── flow diagrams/                    # Pipeline and architecture flow diagrams
└── documents/                        # Supporting documentation and figures
```

---

## Models

### ⭐ Proposed Architecture: DHCANet

**DHCANet (Dual-stream Hierarchical Channel Attention Network)** is the novel architecture proposed in this study. It is implemented in [`autism-code-binary-detail.ipynb`](autism-code-binary-detail.ipynb).

| Property | Details |
|----------|---------|
| **Architecture** | Dual-stream hybrid (DenseNet201 + MobileNetV2) with custom attention pyramid |
| **Total Parameters** | 22,912,698 |
| **Trainable Parameters** | 2,595,898 |
| **Input Resolution** | 224 × 224 |
| **Test Accuracy** | **93.67%** |
| **AUC-ROC** | **0.9887** |
| **Cohen's Kappa** | 0.8733 |
| **Matthews Corrcoef** | 0.8744 |

DHCANet consists of three parallel feature extraction streams:

1. **DenseNet201 stream** — Pretrained ImageNet backbone (frozen) for dense hierarchical feature extraction.
2. **MobileNetV2 stream** — Pretrained ImageNet backbone (frozen) serving as an efficient Xception-style feature extractor.
3. **Custom Attention Pyramid** — A multi-stage feature pyramid with novel attention modules:
   - **HMCAM (Hierarchical Multi-Head Channel Attention Module)**: Squeeze-and-excitation with multi-head attention for adaptive channel recalibration.
   - **TDCAM (Temporal-Depth Convolutional Attention)**: Residual blocks combining spatial and channel-wise feature extraction with HMCAM gating.
   - **IRMCA (Inverted Residual Mobile Convolution with Attention)**: Efficient depthwise separable convolutions with attention gating and residual connections.

Features from all three streams are fused via concatenation and passed through a multi-layer classifier with dropout regularization.

---

### Benchmark Models

The following **33 deep learning architectures** are systematically benchmarked in this study:

### Convolutional Neural Networks (CNNs)

| Model | Parameters | Notebook |
|-------|-----------|----------|
| ResNet-18 | ~11.7M | `resnet18-vgg16-19-effnet-b0-b1-b2.ipynb` |
| VGG-16 | ~138.4M | `resnet18-vgg16-19-effnet-b0-b1-b2.ipynb` |
| VGG-19 | ~143.7M | `resnet18-vgg16-19-effnet-b0-b1-b2.ipynb` |
| EfficientNet-B0 | ~5.3M | `resnet18-vgg16-19-effnet-b0-b1-b2.ipynb` |
| EfficientNet-B1 | ~7.8M | `resnet18-vgg16-19-effnet-b0-b1-b2.ipynb` |
| EfficientNet-B2 | ~9.2M | `resnet18-vgg16-19-effnet-b0-b1-b2.ipynb` |
| AlexNet | ~61.1M | `vit-alexnet-mobilenet-densenet-regnet.ipynb` |
| DenseNet-121 | ~8.0M | `vit-alexnet-mobilenet-densenet-regnet.ipynb` |
| DenseNet-169 | ~14.1M | `vit-alexnet-mobilenet-densenet-regnet.ipynb` |
| DenseNet-201 | ~20.0M | `densenet-201-xceptionnet.ipynb` |
| MobileNet-V2 | ~3.5M | `vit-alexnet-mobilenet-densenet-regnet.ipynb` |
| MobileNet-V3-Large | ~5.5M | `vit-alexnet-mobilenet-densenet-regnet.ipynb` |
| RegNet-X-400MF | ~5.5M | `vit-alexnet-mobilenet-densenet-regnet.ipynb` |
| XceptionNet | ~22.9M | `densenet-201-xceptionnet.ipynb` |

### Vision Transformers & Hybrid Architectures

| Model | Parameters | Notebook |
|-------|-----------|----------|
| ViT-B/16 | ~86.6M | `vit-alexnet-mobilenet-densenet-regnet.ipynb` |
| DeiT-Tiny | ~5.7M | `deit-swin-convnext-clip-mvit.ipynb` |
| DeiT-Small | ~22.1M | `deit-swin-convnext-clip-mvit.ipynb` |
| DeiT-Base | ~86.6M | `deit-swin-convnext-clip-mvit.ipynb` |
| Swin-Tiny | ~28.3M | `deit-swin-convnext-clip-mvit.ipynb` |
| Swin-Small | ~49.6M | `deit-swin-convnext-clip-mvit.ipynb` |
| Swin-Base | ~87.8M | `deit-swin-convnext-clip-mvit.ipynb` |
| ConvNeXt-Tiny | ~28.6M | `deit-swin-convnext-clip-mvit.ipynb` |
| ConvNeXt-Small | ~50.2M | `deit-swin-convnext-clip-mvit.ipynb` |
| ConvNeXt-Base | ~88.6M | `deit-swin-convnext-clip-mvit.ipynb` |
| MViT-Tiny | ~24.0M | `deit-swin-convnext-clip-mvit.ipynb` |
| MViT-Small | ~34.9M | `deit-swin-convnext-clip-mvit.ipynb` |
| MViT-Base | ~51.4M | `deit-swin-convnext-clip-mvit.ipynb` |
| CLIP-ViT-B/32 | ~88.2M | `deit-swin-convnext-clip-mvit.ipynb` |

### Face Recognition Networks

| Model | Parameters | Notebook |
|-------|-----------|----------|
| FaceNet | ~23.5M | `facenet-insightface-vggface2.ipynb` |
| InsightFace | ~25.0M | `facenet-insightface-vggface2.ipynb` |
| VGGFace2 | ~25.0M | `facenet-insightface-vggface2.ipynb` |
| ArcFace | ~25.0M | `cosface-arcface.ipynb` |
| CosFace | ~25.0M | `cosface-arcface.ipynb` |

---

## Dataset

### Description

The study utilizes a curated **facial image dataset** for binary classification of Autism Spectrum Disorder:

| Property | Details |
|----------|---------|
| **Classes** | 2 — `Autistic`, `Non_Autistic` |
| **Training samples** | ~2,540 images (balanced) |
| **Validation samples** | Stratified split |
| **Test samples** | Stratified split |
| **Image format** | JPEG |
| **Preprocessing** | Face detection, alignment, deduplication, quality filtering |

### Data Pipeline

The data preprocessing pipeline consists of the following stages:

1. **Data Collection**: Aggregation of facial images from multiple publicly available sources.
2. **Data Cleaning** (`part 1/data-cleaning.ipynb`): Removal of corrupted images, resolution filtering, and format standardization.
3. **Deduplication** (`part 1/data-cleaning.ipynb`): Perceptual hash–based duplicate and near-duplicate removal to prevent data leakage.
4. **Dataset Splitting** (`part 1/data-preparation.ipynb`): Stratified train/validation/test splitting with class balance preservation.
5. **Exploratory Analysis** (`part 1/overview.ipynb`): Class distribution analysis, image dimension statistics, and sample visualization.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training; CPU inference is supported)
- pip or conda package manager

### Dependencies

```bash
# Clone the repository
git clone https://github.com/ronakrpanchal/ASD-Research.git
cd ASD-Research

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0.0 | Deep learning framework |
| `torchvision` | ≥ 0.15.0 | Pretrained models & image transforms |
| `timm` | ≥ 0.9.0 | Transformer and advanced CNN architectures |
| `numpy` | ≥ 1.24.0 | Numerical computing |
| `pandas` | ≥ 2.0.0 | Data manipulation & analysis |
| `matplotlib` | ≥ 3.7.0 | Visualization |
| `seaborn` | ≥ 0.12.0 | Statistical visualization |
| `scikit-learn` | ≥ 1.2.0 | Metrics, preprocessing, evaluation |
| `Pillow` | ≥ 9.5.0 | Image loading & processing |
| `opencv-python` | ≥ 4.7.0 | Computer vision utilities |
| `grad-cam` | ≥ 1.4.0 | Grad-CAM visualizations |
| `shap` | ≥ 0.42.0 | SHAP explainability |
| `lime` | ≥ 0.2.0 | LIME explanations |
| `facenet-pytorch` | ≥ 2.5.0 | FaceNet implementation |
| `openpyxl` | ≥ 3.1.0 | Excel file I/O |
| `jupyter` | ≥ 1.0.0 | Notebook execution |
| `tqdm` | ≥ 4.65.0 | Progress bars |

> **Note:** If you do not have a `requirements.txt` yet, you may install packages individually:
> ```bash
> pip install torch torchvision timm numpy pandas matplotlib seaborn scikit-learn Pillow opencv-python grad-cam shap lime facenet-pytorch openpyxl jupyter tqdm
> ```

---

## Usage

### 1. Data Preparation

```bash
# Run the data cleaning notebook
jupyter notebook "part 1/data-cleaning.ipynb"

# Run the data preparation notebook
jupyter notebook "part 1/data-preparation.ipynb"
```

### 2. Model Training

#### Proposed Model (DHCANet)

To train the final proposed DHCANet model:

```bash
# Train DHCANet (Dual-stream Hierarchical Channel Attention Network)
jupyter notebook "autism-code-binary-detail.ipynb"
```

**DHCANet training configuration:**
- **Optimizer:** Adam (lr=5e-5)
- **Loss function:** CrossEntropyLoss
- **Mixed precision:** Enabled (torch.cuda.amp)
- **Gradient clipping:** max_norm=1.0
- **Scheduler:** ReduceLROnPlateau (mode='max', factor=0.2, patience=3)
- **Epochs:** Up to 15 with early stopping at >93% validation accuracy
- **Backbone freezing:** DenseNet201 and MobileNetV2 backbones are frozen; only custom attention modules and classifier are trained

#### Benchmark Models

Each benchmark model family is organized into separate Jupyter notebooks. To train a specific group of models:

```bash
# Example: Train CNN models (ResNet-18, VGG-16/19, EfficientNet-B0/B1/B2)
jupyter notebook "Model Training and Evaluation/resnet18-vgg16-19-effnet-b0-b1-b2.ipynb"

# Example: Train Vision Transformers (DeiT, Swin, ConvNeXt, CLIP, MViT)
jupyter notebook "Model Training and Evaluation/deit-swin-convnext-clip-mvit.ipynb"

# Example: Train face recognition models (FaceNet, InsightFace, VGGFace2)
jupyter notebook "Model Training and Evaluation/facenet-insightface-vggface2.ipynb"
```

**Training configuration (common across benchmark models):**
- **Optimizer:** Adam / AdamW
- **Loss function:** CrossEntropyLoss
- **Learning rate:** Model-specific (documented in each notebook)
- **Batch size:** Model-specific based on GPU memory
- **Epochs:** Early stopping with patience monitoring
- **Data augmentation:** Random horizontal flip, random rotation, color jitter, normalization
- **Transfer learning:** ImageNet-pretrained weights with fine-tuning

### 3. Evaluation

Trained model checkpoints are saved in the `Model results/` directory. To evaluate:

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load a trained model (example: EfficientNet-B0)
model = models.efficientnet_b0(pretrained=False, num_classes=2)
model.load_state_dict(torch.load("Model results/best_effnet-b0.pth", map_location="cpu"))
model.eval()

# Preprocess input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("path/to/facial_image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    label = "Autistic" if prediction == 0 else "Non_Autistic"
    print(f"Prediction: {label}")
```

### 4. Qualitative Analysis (Interpretability)

#### Grad-CAM Visualization

```bash
# Generate Grad-CAM heatmaps for specific models
jupyter notebook "Qualitative Analysis/GradCam/Training/gradcam.ipynb"
```

Grad-CAM highlights the discriminative facial regions used by each model for classification. Results are saved in `Qualitative Analysis/GradCam/Results/`.

#### SHAP & LIME Explanations

```bash
# Generate SHAP/LIME explanations (example: DenseNet-201)
jupyter notebook "Qualitative Analysis/SHAP_Lime/Training/shap-lime-densenet-201.ipynb"
```

SHAP and LIME provide complementary model-agnostic explanations for individual predictions. Separate notebooks are provided for each of the 20+ evaluated models.

### 5. Metrics Extraction

```bash
# Extract and aggregate performance metrics across all models
jupyter notebook "part 1/model_metrics_extractor.ipynb"
```

Aggregated results are exported to:
- `Results/ASD_Model_Performance_Metrics_Merged.xlsx`
- `Results/ASD_Model_Performance_Metrics_Simple.xlsx`

---

## Results

All models are evaluated on the held-out test set using the following metrics:

- **Accuracy** — Overall classification accuracy
- **Precision** — Positive predictive value per class
- **Recall (Sensitivity)** — True positive rate per class
- **F1-Score** — Harmonic mean of precision and recall
- **AUC-ROC** — Area Under the Receiver Operating Characteristic curve
- **Cohen's Kappa** — Inter-rater agreement adjusted for chance
- **Matthews Correlation Coefficient (MCC)** — Balanced measure for binary classification

### DHCANet (Proposed Model) — Test Set Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **93.67%** |
| **Test Loss** | 0.1734 |
| **AUC-ROC** | **0.9887** |
| **Balanced Accuracy** | 0.9367 |
| **Cohen's Kappa** | 0.8733 |
| **Matthews Corrcoef** | 0.8744 |

**Per-class performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Autistic | 0.9781 | 0.8933 | 0.9338 | 150 |
| Non-Autistic | 0.9018 | 0.9800 | 0.9393 | 150 |
| **Macro Avg** | **0.9400** | **0.9367** | **0.9365** | 300 |

### Benchmark Results

Detailed per-model results, comparison tables, and analysis are provided in:
- `Results/ASD_Model_Performance_Metrics_Merged.xlsx`
- LaTeX tables in `part 1/tex files/`
- PDF summaries in `part 1/model tables/`

> **📊 For complete quantitative results and analysis, please refer to the manuscript.**

---

## Key Algorithms & Implementation Details

### DHCANet Architecture (Proposed)

DHCANet employs a **three-stream feature fusion** strategy with the following novel components:

#### HMCAM (Hierarchical Multi-Head Channel Attention Module)
Implements squeeze-and-excitation with multi-head attention for adaptive channel recalibration:
```python
# Global Average Pooling → FC → ReLU → FC → Sigmoid → Channel Reweighting
squeeze = GAP(x)  # [B, C, 1, 1]
excitation = sigmoid(fc2(relu(fc1(squeeze))))  # [B, C, 1, 1]
output = x * excitation  # Channel-wise attention
```

#### TDCAM (Temporal-Depth Convolutional Attention)
Residual blocks combining spatial convolutions with HMCAM gating:
- 3×3 convolution → BatchNorm → ReLU → HMCAM → 1×1 convolution → Residual connection

#### IRMCA (Inverted Residual Mobile Convolution with Attention)
Efficient depthwise separable convolutions with attention:
- 1×1 expand → Depthwise 3×3 → BatchNorm → ReLU → HMCAM → 1×1 project → Residual connection

#### Feature Fusion & Classification
```
DenseNet201 (1920-d) + MobileNetV2 (1280-d) + Pyramid (256-d) = 3456-d
→ FC(512) → ReLU → Dropout(0.5)
→ FC(256) → ReLU → Dropout(0.4)
→ FC(128) → ReLU → Dropout(0.3)
→ FC(2) → Output
```

### Transfer Learning Strategy

All benchmark models are initialized with **ImageNet-pretrained weights** and fine-tuned on the ASD facial image dataset. The final classification layer is replaced with a 2-class linear head. Both feature extraction (frozen backbone) and full fine-tuning strategies are explored.

For DHCANet, the DenseNet201 and MobileNetV2 backbones are **frozen** during training, while only the custom attention pyramid and classifier head are trained — resulting in only **2.6M trainable parameters** out of 22.9M total.

### Data Augmentation Pipeline

```python
# Benchmark models
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DHCANet (no augmentation — clean evaluation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Model Interpretability

- **Grad-CAM (Gradient-weighted Class Activation Mapping):** Produces coarse localization heatmaps highlighting the image regions most relevant to the model's prediction, computed via gradients of the target class flowing into the final convolutional layer.
- **SHAP (SHapley Additive exPlanations):** Provides theoretically grounded, additive feature attribution scores based on cooperative game theory, applied at the superpixel level for image inputs.
- **LIME (Local Interpretable Model-agnostic Explanations):** Generates locally faithful explanations by perturbing input superpixels and fitting an interpretable surrogate model.

---

## Reproducibility

To ensure full reproducibility of reported results:

1. **Environment**: Use the specified package versions listed in the [Dependencies](#core-dependencies) section.
2. **Random seeds**: All notebooks set deterministic seeds for PyTorch, NumPy, and Python's random module.
3. **Dataset**: Use the preprocessed dataset in `AutismDataset/` or regenerate from source using the data pipeline notebooks.
4. **Hardware**: Experiments were conducted on NVIDIA GPUs. Minor floating-point variations may occur across different GPU architectures.
5. **Model weights**: Pretrained model checkpoints are available in `Model results/` for direct evaluation without retraining.

---

## Citation

If you use this code, data, or any materials from this repository in your research, **please cite the following manuscript**:

```bibtex
@article{panchal2025asd,
  title     = {Autism Spectrum Disorder Detection via Facial Image Analysis: A Comprehensive Benchmarking of Deep Learning Architectures},
  author    = {Panchal, Ronak and others},
  journal   = {The Visual Computer},
  year      = {2025},
  publisher = {Springer},
  note      = {Under Review}
}
```

> **📌 This repository is the official code companion to the above manuscript submitted to *The Visual Computer*. Readers and researchers are encouraged to cite this manuscript when referencing this work.**

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We acknowledge the creators of the public facial image datasets used in this study, as well as the open-source deep learning community for providing pretrained model architectures.

---

## Contact

For questions, issues, or collaboration inquiries related to this repository or the associated manuscript, please:

- **Open an issue** on this repository
- **Contact the corresponding author** via the information provided in the manuscript

---

<p align="center">
  <sub>📝 This repository accompanies a manuscript submitted to <b>The Visual Computer</b> (Springer). Please cite accordingly.</sub>
</p>
