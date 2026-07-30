## First Approach

- Use ResNet18 models with default weights and freeze parameters
- Check the accuracy with 5 epochs

```Output
[info] Loaded dataset with 37131 samples
[info] Split sizes -> train: 25991, val: 7426, test: 3714
[info] DataLoader setup -> batch_size=32, num_workers=0, pin_memory=False
[info] Class counts -> gender=5, article=29, color=39, usage=6

[info] Backbone -> resnet18, freeze_backbone=True
[info] Training on device: cpu
[info] Starting training and validation loop...

Epoch 1/5
Train Loss      : 3.4183
Validation Loss : 2.7116
Train Accuracy  : gender=78.85%, article=74.08%, color=52.17%, usage=84.25%
Val Accuracy    : gender=83.03%, article=81.00%, color=58.86%, usage=86.82%
------------------------------------------------------------
Epoch 2/5
Train Loss      : 2.7568
Validation Loss : 2.5591
Train Accuracy  : gender=82.71%, article=81.05%, color=57.77%, usage=86.60%
Val Accuracy    : gender=82.99%, article=83.89%, color=61.08%, usage=87.64%
------------------------------------------------------------
Epoch 3/5
Train Loss      : 2.5905
Validation Loss : 2.4394
Train Accuracy  : gender=84.24%, article=82.42%, color=59.48%, usage=87.45%
Val Accuracy    : gender=85.83%, article=84.70%, color=61.49%, usage=88.24%
------------------------------------------------------------
Epoch 4/5
Train Loss      : 2.4625
Validation Loss : 2.3022
Train Accuracy  : gender=85.45%, article=83.47%, color=60.32%, usage=88.04%
Val Accuracy    : gender=86.18%, article=85.50%, color=63.08%, usage=89.38%
------------------------------------------------------------
Epoch 5/5
Train Loss      : 2.3674
Validation Loss : 2.3331
Train Accuracy  : gender=86.03%, article=84.24%, color=61.14%, usage=88.65%
Val Accuracy    : gender=87.13%, article=84.80%, color=61.94%, usage=89.33%
------------------------------------------------------------
```

### Observations (Achievements)

- The training and validation loop looks good! We are able to calculate the loss and accuracy
- The validation accuracy is greater than train accuracy that means, the dropout is doing its job.
- The gender, article and usage accuracy is Ok (not good, Target is > 90%)

### Observations (Need improvement)

- But the color accuracy is Bad, its just 60%. Reason: The color data is getting loss in going forward for the already trained model.
- That means the color tensors are not getting trained properly

#### Things to try for improvements

- Lets unfreeze the paramters and let the paramters get trained. and observe the accuracy again!

## Second Approach

- Use ResNet18 model with default weights and unfreeze parameters
- Check the accuracy with 5 epochs

```
[info] Loaded dataset with 37131 samples
[info] Split sizes -> train: 25991, val: 7426, test: 3714
[info] DataLoader setup -> batch_size=32, num_workers=0, pin_memory=False
[info] Class counts -> gender=5, article=29, color=39, usage=6

[info] Backbone -> resnet18, freeze_backbone=False
[info] Training on device: cpu
[info] Starting training and validation loop...

Epoch 1/5
Train Loss : 2.9855
Validation Loss : 2.4136
Train Accuracy : gender=83.68%, article=77.81%, color=56.01%, usage=87.16%
Val Accuracy : gender=87.75%, article=84.47%, color=61.82%, usage=89.77%
------------------------------------------------------------
Epoch 2/5
Train Loss : 2.2173
Validation Loss : 2.0498
Train Accuracy : gender=88.49%, article=86.55%, color=64.25%, usage=89.94%
Val Accuracy : gender=90.04%, article=87.09%, color=66.15%, usage=90.30%
------------------------------------------------------------
Epoch 3/5
Train Loss : 1.9606
Validation Loss : 1.9353
Train Accuracy : gender=90.13%, article=88.98%, color=66.52%, usage=91.00%
Val Accuracy : gender=90.63%, article=90.36%, color=67.83%, usage=91.14%
------------------------------------------------------------
Epoch 4/5
Train Loss : 1.7749
Validation Loss : 1.9892
Train Accuracy : gender=91.12%, article=90.75%, color=68.41%, usage=91.86%
Val Accuracy : gender=90.82%, article=87.25%, color=67.63%, usage=90.65%
------------------------------------------------------------
Epoch 5/5
Train Loss : 1.6149
Validation Loss : 1.8651
Train Accuracy : gender=92.05%, article=91.86%, color=70.70%, usage=92.70%
Val Accuracy : gender=91.88%, article=89.28%, color=68.84%, usage=91.62%
```
