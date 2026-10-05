# Waste Classification for Smart Recycling

Deep-learning project for classifying waste images into five categories:
**plastic, paper, glass, metal, and organic waste**.

## Project goal

Build and compare a baseline custom Convolutional Neural Network (CNN) with a
MobileNetV2 transfer-learning model. The project will select the model with the
strongest held-out test performance for a smart-recycling image-classification demo.

## Dataset plan

The selected source is Kaggle's **Garbage Classification** dataset. Its labels
are standardized into the five classes required by the assignment:

- `plastic`
- `paper`
- `glass`
- `metal`
- `organic`

The source labels `paper`, `plastic`, and `metal` are retained unchanged.
`biological` is mapped to `organic`, while `white-glass`, `green-glass`, and
`brown-glass` are merged into `glass`. The remaining source labels are excluded
so every prediction matches the assignment specification.

Current standardized dataset counts before the corruption audit:

| Class | Images |
| --- | ---: |
| Plastic | 865 |
| Paper | 1,050 |
| Glass | 2,011 |
| Metal | 769 |
| Organic | 985 |
| **Total** | **5,680** |

Dataset source: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

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
