# Phase 2: Split and preprocessing

The audited image manifest is split with a fixed random seed (`42`) into:

| Split | Purpose | Proportion |
| --- | --- | ---: |
| Training | Learns model parameters | 70% |
| Validation | Tunes the model and detects overfitting | 15% |
| Test | Final, unbiased comparison of models | 15% |

The split is **stratified**, so every split preserves the original class
distribution. The test set is never used to choose model settings.

Images are decoded as RGB, resized to `224 × 224`, and emitted in batches of
32. Pixel normalization and image augmentation will be embedded in the training
models, which prevents augmentation from affecting validation or test images.

Because the glass class has more images than the other categories, the training
stage uses class weights computed only from the training split.
