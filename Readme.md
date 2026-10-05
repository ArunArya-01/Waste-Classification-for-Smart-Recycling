# Waste Classification for Smart Recycling

Deep-learning project for classifying waste images into five categories:
**plastic, paper, glass, metal, and organic waste**.

## Project goal

Build and compare a baseline custom Convolutional Neural Network (CNN) with a
MobileNetV2 transfer-learning model. The project will select the model with the
strongest held-out test performance for a smart-recycling image-classification demo.

## Dataset plan

The selected source is the **Merged Waste Classification Dataset (MWCD)**, a
CC BY 4.0 dataset containing 35,168 RGB images across standard waste labels.
This project uses only these five labels:

- `plastic`
- `paper`
- `glass`
- `metal`
- `organic`

Other MWCD labels are deliberately excluded so every prediction matches the
assignment specification. The exact image count, class balance, corrupt-file
check, and final split counts will be generated in the Colab dataset-audit step.

Dataset source: https://data.mendeley.com/datasets/x863v66fv3/1

## Planned workflow

1. Audit and split the selected dataset into train, validation, and test sets.
2. Preprocess images and apply training-only augmentation.
3. Train a baseline custom CNN.
4. Train and fine-tune MobileNetV2.
5. Compare accuracy, precision, recall, F1-score, confusion matrices, and learning curves.
6. Demonstrate the selected model with an image-upload prediction interface.

## Repository layout

```text
data/       # ignored raw/processed datasets
notebooks/  # Colab implementation notebook
src/        # reusable Python modules
app/        # optional Streamlit prediction demo
models/     # ignored trained-model files
outputs/    # saved evaluation figures and sample predictions
docs/       # presentation/report material
```

## Running training

Training will run in Google Colab with GPU acceleration to avoid sustained
load and heat on the local MacBook. The source code and lightweight outputs are
kept in Git; raw datasets and model checkpoint files are ignored.
