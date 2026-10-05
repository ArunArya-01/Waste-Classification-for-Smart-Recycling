# Dataset description

## Source

Kaggle: [Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification).

## Label standardization

The original dataset has 12 household-waste classes. To match the assignment,
the project retains or maps labels as follows:

| Original label | Project label | Reason |
| --- | --- | --- |
| `plastic` | Plastic | Direct material category |
| `paper` | Paper | Direct material category |
| `metal` | Metal | Direct material category |
| `biological` | Organic | Food/biological waste is treated as organic waste |
| `white-glass`, `green-glass`, `brown-glass` | Glass | Glass colour is irrelevant to the assignment's material-level label |

All other source labels are excluded. This makes the model a focused,
five-class classifier rather than incorrectly presenting it as able to predict
categories that were not requested.

## Initial class counts

| Class | Image count |
| --- | ---: |
| Plastic | 865 |
| Paper | 1,050 |
| Glass | 2,011 |
| Metal | 769 |
| Organic | 985 |
| **Total** | **5,680** |

The classes are imbalanced, particularly glass. Training will use class weights
and training-only image augmentation; the validation and test sets will remain
unaugmented.
