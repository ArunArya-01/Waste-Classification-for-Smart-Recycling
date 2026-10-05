# Baseline custom CNN

## Purpose

This model is the project baseline. Its results will be compared with the
MobileNetV2 transfer-learning model in the final analysis.

## Architecture

```text
224 × 224 RGB image
  → training-only augmentation
  → rescaling (pixel values / 255)
  → Conv2D(32) + max pooling
  → Conv2D(64) + batch normalization + max pooling
  → Conv2D(128) + batch normalization + max pooling
  → Conv2D(256) + batch normalization
  → global average pooling
  → dropout(0.40)
  → dense(128, ReLU)
  → dropout(0.25)
  → dense(5, softmax)
```

The softmax output produces probabilities for plastic, paper, glass, metal,
and organic waste. Data augmentation, dropout, batch normalization, early
stopping, learning-rate reduction, and class weights are used to improve
generalization and reduce class-imbalance bias.
