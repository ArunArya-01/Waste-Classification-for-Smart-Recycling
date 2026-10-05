# Data directory

Raw and processed images are intentionally excluded from Git because they are
large generated/downloaded artifacts.

The Colab setup downloads the public Kaggle Garbage Classification dataset and
standardizes it to five assignment labels: `plastic`, `paper`, `glass`,
`metal`, and `organic`. The original `biological` folder becomes `organic`,
and the three colour-specific glass folders are combined into `glass`.

The audit notebook validates files and records counts before the reproducible
stratified train/validation/test split is created.
