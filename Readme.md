# ♻️ Waste Classification for Smart Recycling

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

An end-to-end Computer Vision system that automatically classifies recyclable and municipal waste images into **five core categories** (**Glass, Metal, Organic, Paper, and Plastic**) to automate municipal sorting, prevent contamination, and power intelligent smart-recycling bins.

The system builds, evaluates, and unites **three distinct modeling strategies**:
1. **Baseline Custom CNN** (Trained from scratch with modern regularization)
2. **MobileNetV2 Transfer Learning** (ImageNet feature extraction + top-layer fine-tuning)
3. **Hybrid Soft-Voting Ensemble** (Adaptive probability blending that combines the strengths of both models for optimal accuracy and robustness)

---

## 📌 Table of Contents

- [Key Features](#-key-features)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Dataset & Preprocessing Pipeline](#-dataset--preprocessing-pipeline)
- [Model Architectures](#-model-architectures)
  - [1. Baseline Custom CNN](#1-baseline-custom-cnn)
  - [2. MobileNetV2 Transfer Learning](#2-mobilenetv2-transfer-learning)
  - [3. Hybrid Probability Ensemble](#3-hybrid-probability-ensemble)
- [Performance & Model Comparison](#-performance--model-comparison)
- [Visualizations & Evaluation Gallery](#-visualizations--evaluation-gallery)
- [Smart Recycling Bin & Action Matrix](#-smart-recycling-bin--action-matrix)
- [Interactive Demo Web App](#-interactive-demo-web-app)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)

---

## 🌟 Key Features

* **5-Class Standardized Waste Segregation:** Strictly mapped to municipal recycling requirements (**Glass**, **Metal**, **Organic**, **Paper**, **Plastic**).
* **Multi-Stage Modeling Pipeline:** From scratch baseline CNN to deep transfer learning and hybrid ensemble fusion.
* **Class-Imbalance Mitigation:** Incorporates computed heuristic class weighting to prevent dominant classes (e.g., glass) from biasing predictions.
* **Probability Soft-Voting Ensemble:** Dynamic grid-search optimization to identify the ideal blending weight ratio between specialized architectures.
* **Disagreement & Error Analysis:** Automated diagnostics analyzing samples where individual models diverge, demonstrating how the ensemble resolves conflicts.
* **Interactive Smart-Recycling UI:** Streamlit interface that accepts image uploads or camera input, predicts waste category with class-by-class confidence scores, and provides immediate disposal instructions and bin color coding.

---

## 🏗️ System Architecture & Workflow

```text
Input Waste Image (224 × 224 × 3)
                │
                ▼
   ┌──────────────────────────┐
   │ Preprocessing & Scaling  │
   │  (tf.image normalization) │
   └────────────┬─────────────┘
                │
        ┌───────┴────────────────────────┐
        ▼                                ▼
┌───────────────────────────┐    ┌───────────────────────────┐
│     Custom 4-Block CNN    │    │    MobileNetV2 Backbone   │
│   (Scratch Feature Map)   │    │ (ImageNet Feature Extractor)│
└─────────────┬─────────────┘    └─────────────┬─────────────┘
              │ Softmax Probabilities         │ Softmax Probabilities
              │ P_cnn (5 classes)             │ P_mobilenet (5 classes)
              └───────────────┬───────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │    Hybrid Soft-Voting       │
               │      Probability Fusion     │
               │   P = (1-w)·P_cnn + w·P_mob │
               └──────────────┬──────────────┘
                              │
                              ▼
               ┌─────────────────────────────┐
               │    Prediction & Guidance    │
               │  • Predicted Waste Class    │
               │  • Confidence Distribution  │
               │  • Smart Bin Color & Rules  │
               └─────────────────────────────┘
```

---

## 📊 Dataset & Preprocessing Pipeline

The project utilizes standardized imagery from Kaggle's **Garbage Classification** dataset, filtered and audited to eliminate corruption and map directly into standard recycling categories.

### 1. Label Mapping

| Original Source Label | Standardized Class | Municipal Rationale |
|---|---|---|
| `glass`, `white-glass`, `green-glass`, `brown-glass` | **Glass** | Color sub-types consolidated into single material stream |
| `metal` | **Metal** | Cans, tins, foil, and metallic containers |
| `biological` | **Organic** | Food scraps, fruit peels, compostable matter |
| `paper`, `cardboard` | **Paper** | Clean paper, cardboard packaging, cartons |
| `plastic` | **Plastic** | Bottles, polymer containers, plastic packaging |

### 2. Dataset Distribution & Splitting

The audited dataset is partitioned using **stratified random splitting** (fixed seed `42`), preserving class ratios across all splits:

| Split | Proportion | Purpose |
|---|:---:|---|
| **Training Set** | **70%** | Feature learning and backpropagation |
| **Validation Set** | **15%** | Hyperparameter tuning, EarlyStopping, and checkpoint selection |
| **Test Set** | **15%** | Untouched final evaluation and unbiased model benchmark |

* **Class Weighting:** Applied during training to balance loss contribution inversely proportional to class frequencies.
* **Augmentation Pipeline (Training-Only):**
  * Horizontal Flipping
  * Random Rotation ($\pm 10\%$)
  * Random Zoom ($\pm 10\%$)
  * Random Contrast Adjustment ($\pm 10\%$)

---

## 🧠 Model Architectures

### 1. Baseline Custom CNN
Designed to establish a benchmark for learning representations strictly from scratch:
* **Feature Extractor:** 4 sequential blocks of `Conv2D` ($3 \times 3$ kernels: 32, 64, 128, 256 filters) coupled with `BatchNormalization` and `MaxPooling2D`.
* **Spatial Compression:** `GlobalAveragePooling2D` to reduce parameter count and combat spatial overfitting.
* **Regularization & Head:** Dense layer (128 units, ReLU) with multi-stage Dropout ($0.40$ and $0.25$) feeding a 5-unit Softmax output.

### 2. MobileNetV2 Transfer Learning
Leverages inverted residual bottlenecks and depthwise separable convolutions pre-trained on ImageNet (1.4M images):
* **Backbone:** Pretrained MobileNetV2 feature extractor initialized with frozen ImageNet weights.
* **Phase 1 (Head Training):** Trained custom classification top (`GlobalAveragePooling2D` $\to$ `Dropout(0.30)` $\to$ `Dense(5, Softmax)`) with Adam ($\text{lr} = 10^{-3}$).
* **Phase 2 (Deep Fine-Tuning):** Unfroze top 30 layers of the backbone with fine-grain Adam learning rate ($\text{lr} = 10^{-5}$) to specialize visual edge filters to waste items.

### 3. Hybrid Probability Ensemble
Unites the distinct representations of both models via weighted soft-voting:
$$\hat{P}(y = c \mid \mathbf{x}) = (1 - w) \cdot P_{\text{CNN}}(y = c \mid \mathbf{x}) + w \cdot P_{\text{MobileNet}}(y = c \mid \mathbf{x})$$

* A systematic grid search across $w \in [0.0, 1.0]$ pinpoints the exact mathematical balance between general domain transfer knowledge and dataset-specific convolutional filters.

---

## 📈 Performance & Model Comparison

*Evaluated on the completely held-out, untouched test split (852 images, 15% of dataset):*

| Model Architecture | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Remarks |
|---|:---:|:---:|:---:|:---:|---|
| **Custom CNN (Baseline)** | 64.32% | 68.51% | 64.32% | 63.50% | Scratch baseline (struggles with complex transparency/shapes) |
| **Combined Ensemble (Equal 50/50)** | 90.38% | 90.83% | 90.38% | 90.32% | Equal blend pulls performance down toward CNN baseline |
| **MobileNetV2 (Fine-Tuned)** | 92.61% | 92.90% | 92.61% | 92.65% | Strong transfer learning feature extraction |
| **Combined Ensemble (Optimal 90/10)** 🏆 | **92.72%** | **93.05%** | **92.72%** | **92.77%** | **Best overall** ($0.90 \times \text{MobileNet} + 0.10 \times \text{CNN}$) |

### 🎯 Per-Class Performance Breakdown (Best Combined Model)

| Waste Category | Precision | Recall | F1-Score | Test Support | Key Characteristic |
|---|:---:|:---:|:---:|:---:|---|
| **Organic** | **97.4%** | **100.0%** | **98.7%** | 148 | Flawless recall: zero organic items missed |
| **Paper** | **97.4%** | 93.0% | **95.1%** | 158 | Exceptional precision across paper & cardboard |
| **Glass** | **95.7%** | 89.0% | **92.3%** | 301 | High precision; minor confusion with clear plastics |
| **Metal** | 84.4% | **93.9%** | **88.9%** | 115 | High recall; catches almost all cans and foil containers |
| **Plastic** | 84.4% | **91.5%** | **87.8%** | 130 | Solid detection; occasional overlap with transparent glass |
| **Overall Weighted** | **93.05%** | **92.72%** | **92.77%** | **852** | **Production-grade waste segregation performance** |

> 💡 *Full evaluation metrics, classification reports, and test arrays are exported in [`outputs/phase6_combined_model/combined_model_comparison.csv`](file:///Users/arunarya/Documents/Waste%20Classification%20for%20Smart%20Recycling/outputs/phase6_combined_model/combined_model_comparison.csv).*

---

## 🖼️ Visualizations & Evaluation Gallery

All diagnostic plots and evaluation figures are automatically generated by the training and evaluation pipelines:

### 1. Training & Convergence Dynamics
Curves tracking training vs. validation accuracy and categorical cross-entropy loss over epochs, showing early stopping points and learning-rate plateaus:
* 📉 **Custom CNN Learning Curves:** `outputs/phase3_custom_cnn/training_curves.png`
* 📉 **MobileNetV2 Fine-Tuning Curves:** `outputs/phase4_mobilenetv2/training_curves.png`

### 2. Confusion Matrix Heatmaps
Detailed visual breakdowns pinpointing classification accuracy and material confusions (e.g., differentiating transparent plastics from clear glass):
* 🔷 **Custom CNN Matrix:** `outputs/phase5_evaluation/custom_cnn_confusion_matrix.png`
* 🔷 **MobileNetV2 Matrix:** `outputs/phase5_evaluation/mobilenetv2_confusion_matrix.png`
* 🟢 **Combined Ensemble Matrix:** `outputs/phase6_combined_model/combined_ensemble_confusion_matrix.png`

### 3. Ensemble Tuning & Disagreement Analysis
* 🎯 **Optimal Blending Curve (`outputs/phase6_combined_model/ensemble_weight_tuning.png`):** Shows accuracy as a function of the MobileNet vs. CNN weight parameter.
* 🔍 **Conflict Resolution Gallery (`outputs/phase6_combined_model/disagreement_analysis.png`):** Visual gallery of challenging test items where individual models offered conflicting predictions, highlighting how probability ensembling arrived at the correct category.

---

## 🗑️ Smart Recycling Bin & Action Matrix

To make the predictions actionable in real-world deployment, the system translates mathematical classifications into concrete recycling instructions:

| Waste Category | Designated Bin Color | Recycling Guidelines & Instructions | Common Examples |
|---|:---:|---|---|
| **Organic** | 🟢 **Green Bin** | Compostable waste. Avoid plastic wraps or stickers. | Fruit peels, leftover food, vegetable waste, tea bags |
| **Paper** | 🔵 **Blue Bin** | Keep dry and clean. Do not mix with oil-soaked cardboard. | Newspapers, cardboard boxes, printer paper, packaging |
| **Glass** | 🟡 **Yellow / Teal Bin** | Rinse containers clean. Remove metal or plastic lids prior to disposal. | Bottles, jars, broken glass containers |
| **Metal** | ⚪ **Gray / Blue Bin** | Rinse thoroughly to eliminate food residue. Flatten cans if possible. | Soda cans, food tins, aluminium foil, metal bottle caps |
| **Plastic** | 🟠 **Orange Bin** | Empty liquids completely, rinse, and check recycling triangle resin code. | PET bottles, milk jugs, detergent containers, food tubs |

---

## 💻 Interactive Demo Web App

An interactive web application built with **Streamlit** allows users to test the trained models in real time:

```bash
# Launch the Streamlit application
streamlit run app/app.py
```

### App Features:
* 📤 **Multi-Format Input:** Drag-and-drop waste images (JPG, PNG, WEBP) or capture directly via webcam.
* 🤖 **Model Selector:** Choose between **MobileNetV2 (High Speed)** or the **Combined Ensemble (Maximum Precision)**.
* 📊 **Confidence Distribution:** Interactive probability breakdown for all 5 candidate waste classes.
* 📋 **Disposal Guidelines:** Real-time recycling instructions and designated bin color indicators based on the detected item.

---

## 📁 Repository Structure

```text
├── README.md                                  # Project overview and documentation
├── requirements.txt                           # Python dependencies
├── app/                                       # Interactive Streamlit prediction application
│   └── app.py                                 # Main application script
├── data/                                      # Data directory (Git-ignored)
│   ├── raw/                                   # Downloaded raw dataset
│   └── processed/                             # Processed and cleaned images
├── docs/                                      # In-depth architectural & technical notes
│   ├── custom_cnn_architecture.md             # Custom CNN layer specifications
│   ├── dataset_description.md                 # Dataset sourcing & category mapping
│   └── phase2_preprocessing.md                # Stratified splitting & augmentation notes
├── models/                                    # Saved model weights (Git-ignored)
│   ├── custom_cnn_best.keras                  # Checkpointed Custom CNN model
│   └── mobilenetv2_best.keras                 # Checkpointed MobileNetV2 model
├── notebooks/                                 # Complete modular Colab/Jupyter workflows
│   ├── 01_phase1_dataset_audit.ipynb          # Dataset inspection & corruption audit
│   ├── 02_phase2_split_and_preprocessing.ipynb# Stratified splitting & tf.data pipeline
│   ├── 03_phase3_custom_cnn_training.ipynb    # Training baseline Custom CNN
│   ├── 04_phase4_mobilenetv2_training.ipynb   # Transfer learning & fine-tuning MobileNetV2
│   ├── 05_phase5_evaluation_and_comparison.ipynb # Comprehensive benchmark & metric analysis
│   └── 06_phase6_combined_ensemble_model.ipynb# Dual-model soft-voting & weight optimization
└── outputs/                                   # Generated figures, reports, and manifests
    ├── phase1_audit/                          # Manifests and corrupt file logs
    ├── phase2_splits/                         # Stratified CSV split files & configurations
    ├── phase3_custom_cnn/                     # CNN training curves and metrics
    ├── phase4_mobilenetv2/                    # MobileNetV2 training curves and metrics
    ├── phase5_evaluation/                     # Individual model confusion matrices & reports
    └── phase6_combined_model/                 # Ensemble tuning curves, matrices, and reports
```

---

## 🚀 Getting Started

### 1. Local Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/ArunArya-01/Waste-Classification-for-Smart-Recycling.git
cd "Waste Classification for Smart Recycling"

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running Training in Google Colab

To leverage GPU acceleration (T4 GPU recommended):
1. Upload or link the repository in **Google Colab**.
2. Mount your Google Drive at `/content/drive/MyDrive/waste-classification`.
3. Execute the notebooks in sequence from [`01_phase1_dataset_audit.ipynb`](notebooks/01_phase1_dataset_audit.ipynb) to [`06_phase6_combined_ensemble_model.ipynb`](notebooks/06_phase6_combined_ensemble_model.ipynb).

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
