# Data directory

Raw and processed images are intentionally excluded from Git because they are
large generated/downloaded artifacts.

The Colab notebook will download or mount the selected dataset, retain only the
five assignment labels (`plastic`, `paper`, `glass`, `metal`, and `organic`),
audit image counts and validity, then create reproducible stratified
train/validation/test splits.
