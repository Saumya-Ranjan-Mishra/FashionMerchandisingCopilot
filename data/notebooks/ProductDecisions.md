## Decision 1

Prediction Heads: Category, Color, Gender, Usage
Reason : These are visually identifiable from a product image and useful for downstream product description generation.

## Decision 2

Drop Season
Reason: Large number of missing values. Cannot be reliably inferred from image.

## Decision 3

Remove article types having fewer than 100 samples.
Reason : Insufficient examples for supervised learning.
