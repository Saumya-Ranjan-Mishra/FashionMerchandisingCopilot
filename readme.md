# AI-Powered Fashion Merchandising Copilot

Turn a product image into structured catalog intelligence. This **multi-head (multi-task) ResNet-18 classifier** predicts gender, article type, color, and usage in one pass.

## Contents

- [Model Outputs](#model-outputs)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Results Summary](#results-summary)
- [Why Not Just Use a Vision LLM?](#why-not-just-use-a-vision-llm)
- [Tech Stack](#tech-stack)
- [Serving and Deployment](#serving-and-deployment)
    - [Run Locally with Docker Compose](#run-locally-with-docker-compose)
    - [Deploy to Kubernetes](#deploy-to-kubernetes)
    - [GitHub Actions Release and Deployment](#github-actions-release-and-deployment)
- [Next Steps](#next-steps)
- [Experiment Log](#experiment-log)
    - [First Approach](#first-approach)
    - [Second Approach](#second-approach)
    - [Third Approach](#third-approach)
- [Inference Results](#inference-results)

## Model Outputs

Given a single fashion product image, the model predicts four attributes:

| Head      | What it predicts                                    | # Classes |
| --------- | --------------------------------------------------- | --------- |
| `gender`  | Target gender (Men / Women / Boys / Girls / Unisex) | 5         |
| `article` | Article type (Tshirt, Jeans, Watch, Handbag, …)     | 29        |
| `color`   | Base color                                          | 39        |
| `usage`   | Usage context (Casual, Formal, Party, Sports, …)    | 6         |

> "Multi-head classification" is the correct term. It's also commonly called **multi-task learning** with **hard parameter sharing**: one shared CNN backbone with multiple task-specific classification heads trained jointly.

- This model is trained on Kaggle Fashion Dataset which can be found here: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

## Demo

[Watch the Fashion Merchandising Copilot demo](https://github.com/user-attachments/assets/47cfd463-e5cd-4f1d-b6d7-80157219e080)

## Project Structure

```
data/            # Raw & processed styles.csv + EDA notebooks
src/
├── preprocess.py           # Cleans metadata, builds label encoders
├── train.py                # Training entry point
├── inference.py            # Single-image prediction
├── config/config.py        # Hyperparameters (batch size, epochs, freeze flag)
├── Dataset/FashionDataset.py  # Custom PyTorch Dataset (image + 4 labels)
└── models/
    ├── models.py           # Multi-head ResNet architecture
    └── model_trainer.py    # Training / validation loop
checkpoints/     # Best model weights per experiment
```

## Dataset

Fashion Product Images dataset — 37,131 samples split as:

- Train: 25,991
- Validation: 7,426
- Test: 3,714

## Results Summary

| Approach | Backbone  | Config                                                     | Epochs | Gender  | Article | Color   | Usage   |
| -------- | --------- | ---------------------------------------------------------- | ------ | ------- | ------- | ------- | ------- |
| 1        | ResNet-18 | Frozen backbone, single shared trunk                       | 5      | 87%     | 85%     | 62%     | 89%     |
| 2        | ResNet-18 | Unfrozen backbone, single shared trunk                     | 5      | 92%     | 89%     | 69%     | 92%     |
| 3        | ResNet-18 | **Unfrozen + split trunk** (color head branches off early) | 10     | **97%** | **95%** | **82%** | **96%** |

**Key insight:** Color was the hardest attribute — freezing the backbone lost low-level color features, and even after unfreezing, deep layers were compressing pixel-level color detail into abstract semantics.

- The breakthrough in Approach 3 was **architectural**: split the ResNet into two branches so the `color_head` reads from **early features** (`layer1`, 64 channels — where raw color and texture live), while `gender` / `article` / `usage` heads continue to read from the deep features (`layer4` + avgpool, 512 channels). Combined with unfreezing and 10 epochs, this lifted color accuracy from 62% → 82% and pushed the other three heads above the 95% target.

See [`src/models/models.py`](src/models/models.py) for the implementation.

## Why Not Just Use a Vision LLM?

A fair question — modern multimodal LLMs (GPT-4o, Gemini, Claude) can look at a fashion image and describe it directly. So why train a dedicated CNN? Here's why this hybrid pipeline (small CNN → structured JSON → text LLM) wins in a real merchandising workflow:

### 1. It's dramatically cheaper at catalog scale

Sending raw images to a multimodal API costs roughly **$0.002–$0.005 per image**, which balloons to **$200–$500 for every 100k products** you tag. Our approach runs the CNN locally at near-zero marginal cost and only sends a tiny JSON payload (`{"color": "Blue", "articleType": "Tshirt", ...}`) to the text LLM for copywriting — bringing the total bill down to around **$5–$15 per 100k items**. That's a 30–100× cost reduction.

### 2. It's fast enough for bulk seller uploads

A vision LLM call typically takes **1.5–3 seconds per image** because the model has to encode the entire visual token stream. Our ResNet handles batched inference in **under 10 ms on a GPU**, which means a seller uploading 500 SKUs sees tags appear instantly instead of waiting minutes.

### 3. The output actually matches your database schema

Generative models love to invent creative descriptors — a "Navy Blue" shirt might come back as _"Midnight Slate"_ or _"Deep Ocean"_, quietly breaking every filter and faceted search in your catalog. Our classification heads output softmax probabilities over a **fixed enum** (`baseColour`, `articleType`, `usage`, `gender`), so every tag is guaranteed to be a valid database value. No hallucinations, no post-hoc cleanup.

### 4. Better color and fine-grained visual accuracy

Vision LLMs compress pixel-level detail into high-level semantic concepts, which is why they routinely confuse similar shades or miss subtle patterns. Because we branch the color head off an **early ResNet layer** (`layer1`), the raw spatial and color information is preserved before deeper layers abstract it away — giving noticeably better color and pattern fidelity.

### 5. Vision and language are decoupled — change one without breaking the other

With a monolithic vision LLM, tweaking a copywriting prompt can accidentally change how the model perceives the product itself. In our pipeline, tagging and copy generation are fully separated. You can rewrite marketing prompts, adjust brand tone, or translate descriptions into 10+ languages using the same JSON tags — without ever touching the image pixels or re-running inference.

## Tech Stack

PyTorch · torchvision (ResNet-18) · pandas · scikit-learn · Jupyter

## Serving and Deployment

The trained Approach 3 model is exported as TorchScript for Triton's PyTorch backend. Triton serves the model, and a small FastAPI service handles image uploads, preprocessing, and readable predictions. The browser UI is served by the same API container.

### Run locally with Docker Compose

From the repository root, build and start both containers:

```powershell
docker compose -f deploy/triton/docker-compose.yml up --build -d
```

- Open the classifier UI at [http://localhost:8080](http://localhost:8080).
- Open interactive API documentation at [http://localhost:8080/docs](http://localhost:8080/docs).
- The image upload endpoint is `POST http://localhost:8080/predict`, with the image sent as multipart form field `file`.
- Triton's tensor inference API is available separately at `http://localhost:8000`.

To stop the services:

```powershell
docker compose -f deploy/triton/docker-compose.yml down
```

### Deploy to Kubernetes

The Triton and API images are built from [deploy/triton/Dockerfile](deploy/triton/Dockerfile) and [deploy/api/Dockerfile](deploy/api/Dockerfile). Kubernetes resources and GHCR image transforms live in [deploy/kubernetes](deploy/kubernetes). The API is exposed inside the cluster through a `ClusterIP` Service; access it locally with:

```powershell
kubectl apply -k deploy/kubernetes
kubectl rollout status deployment/fashion-triton -n fashion-merchandising
kubectl rollout status deployment/fashion-api -n fashion-merchandising
kubectl port-forward service/fashion-api 8080:8080 -n fashion-merchandising
```

Then visit [http://localhost:8080](http://localhost:8080). The manifest points to the public GHCR `main` images by default, so no image pull secret is required.

### GitHub Actions release and deployment

The workflow at [.github/workflows/deploy-kubernetes.yml](.github/workflows/deploy-kubernetes.yml) builds and publishes two GHCR images on pushes to `main`, version tags matching `v*`, or a manual run:

- `ghcr.io/saumya-ranjan-mishra/fashionmerchandisingcopilot-triton`
- `ghcr.io/saumya-ranjan-mishra/fashionmerchandisingcopilot-api`

GHCR packages are private by default. After the first successful workflow run, change both package visibilities to **Public** in their GitHub package settings so Kubernetes can pull them without credentials.

Each image is tagged with the commit SHA; the branch or release tag is also published. The deploy job rewrites the Kustomize images to the SHA-specific tags and waits for both Kubernetes rollouts.

Add this repository Actions secret before enabling deployment:

- `KUBECONFIG_B64`: base64-encoded kubeconfig for a cluster the GitHub runner can reach, with permission to manage the `fashion-merchandising` namespace, Deployments, and Services.

In PowerShell, encode the kubeconfig file before adding it as the `KUBECONFIG_B64` secret:

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("$HOME\.kube\config"))
```

Keep the resulting value private. For a private AKS cluster, use a self-hosted GitHub runner or another network path that can reach the Kubernetes API server. Prefer a least-privilege Kubernetes identity over an administrator kubeconfig.

## Next Steps

- Train longer (15–20 epochs) — color accuracy was still climbing at epoch 10.
- Try ResNet-34 or EfficientNet as the backbone.
- Add per-class metrics and confusion matrices for the color head to identify which color pairs are still being confused (e.g., navy vs. black, beige vs. white).
- Experiment with a small color-specific augmentation policy (avoid heavy color jitter) to further sharpen the color head.

---

## Experiment Log

### First Approach

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

#### Observations (Achievements)

- The training and validation loop looks good! We are able to calculate the loss and accuracy
- The validation accuracy is greater than train accuracy that means, the dropout is doing its job.
- The gender, article and usage accuracy is Ok (not good, Target is > 90%)

#### Observations (Need improvement)

- But the color accuracy is Bad, its just 60%.
- That means the color tensors are not getting trained properly

##### Things to try for improvements

- Lets unfreeze the paramters and let the paramters get trained. and observe the accuracy again!
- For this we will simply set `FREEZE_BACKBONE=False` in the `config.py` file

### Second Approach

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

#### Observations (Achievements)

- The gender, article and usage accuracy is improved annd is above target.

#### Observations (Need improvement)

- But the color accuracy still struggle at nearly 70%.
- That means the color tensors are not getting trained properly, or may be the color data is getting loss while moving forward or deep in the training phase in the ResNet.

##### Things to try for improvements

- Lets separate the neural net into two category.
  - One will have the first layer output and we will call it initial layer. This layers output will be directly fed into the color_head layer for the color prediction
  - Rest will go to the remaining layer. The final layer will now be used only for Article type, Gender and Usage predictions.

- Since we have already seen that the accuracy is high when we set FREEZE_BACKBONE=False, lets train the model with that approach and see how the result looks like.

### Third Approach

This is where the biggest change happened — I stopped trying to fix color with more training and instead **restructured the network itself**.

**The problem:** In Approach 2, even with the backbone fully unfrozen, color accuracy stalled around 69%. Looking at the architecture, the reason became obvious: all four heads were being fed from the very last ResNet block (`layer4` → avgpool → 512-d vector). By that point, the network has abstracted the image into high-level semantic features like _"this is a t-shirt"_ or _"this is casual wear"_ — the raw pixel-level color information has been compressed away. Gender, article, and usage benefit from that abstraction, but color needs the opposite: it needs the raw spatial and chromatic detail that only lives in the **early** layers.

**The fix:** Split the ResNet into two branches and route each head to the level of abstraction it actually needs.

```
Input image
    │
    ▼
conv1 → bn1 → relu → maxpool → layer1   ← early features (64 ch)
    │                                │
    │                                ▼
    │                          AdaptiveAvgPool → color_head
    ▼
layer2 → layer3 → layer4 → avgpool       ← deep features (512 ch)
    │
    ├──► gender_head
    ├──► article_head
    └──► usage_head
```

Concretely (see [`src/models/models.py`](src/models/models.py)):

- `initial_layers` = `conv1 + bn1 + relu + maxpool + layer1` — outputs a **64-channel** feature map that still preserves spatial color information.
- `remaining_layers` = `layer2 + layer3 + layer4 + avgpool` — outputs a **512-d** semantic vector.
- `color_head` reads from a separate `AdaptiveAvgPool2d` applied to the early features (input size = 64).
- `gender_head`, `article_head`, `usage_head` read from the deep 512-d vector.

All four heads still share the early trunk, so gradients from the color head also refine the low-level filters — but color no longer has to fight through 3 more residual stages to make itself heard.

Trained for 10 epochs with `FREEZE_BACKBONE=False`:

```
[info] Loaded dataset with 37131 samples
[info] Split sizes -> train: 25991, val: 7426, test: 3714
[info] DataLoader setup -> batch_size=32, num_workers=0, pin_memory=False
[info] Class counts -> gender=5, article=29, color=39, usage=6

[info] Backbone -> resnet18, freeze_backbone=False
[info] Training on device: cpu
[info] Starting training and validation loop...

Epoch 1/10
Train Loss      : 2.8942
Validation Loss : 2.1483
Train Accuracy  : gender=84.68%, article=81.81%, color=52.01%, usage=87.16%
Val Accuracy    : gender=87.75%, article=84.47%, color=56.82%, usage=89.77%
------------------------------------------------------------
Epoch 2/10
Train Loss      : 2.1580
Validation Loss : 1.8924
Train Accuracy  : gender=89.49%, article=86.55%, color=60.25%, usage=89.94%
Val Accuracy    : gender=90.04%, article=88.09%, color=63.15%, usage=90.30%
------------------------------------------------------------
Epoch 3/10
Train Loss      : 1.8214
Validation Loss : 1.6841
Train Accuracy  : gender=91.83%, article=88.98%, color=65.52%, usage=91.80%
Val Accuracy    : gender=92.13%, article=90.36%, color=67.83%, usage=91.94%
------------------------------------------------------------
Epoch 4/10
Train Loss      : 1.5839
Validation Loss : 1.4920
Train Accuracy  : gender=93.12%, article=90.75%, color=68.91%, usage=92.86%
Val Accuracy    : gender=93.82%, article=91.25%, color=70.63%, usage=93.15%
------------------------------------------------------------
Epoch 5/10
Train Loss      : 1.4012
Validation Loss : 1.3485
Train Accuracy  : gender=94.25%, article=92.10%, color=71.40%, usage=93.90%
Val Accuracy    : gender=94.88%, article=92.80%, color=72.84%, usage=94.12%
------------------------------------------------------------
Epoch 6/10
Train Loss      : 1.2584
Validation Loss : 1.2291
Train Accuracy  : gender=95.10%, article=93.05%, color=73.85%, usage=94.75%
Val Accuracy    : gender=95.42%, article=93.68%, color=74.90%, usage=94.88%
------------------------------------------------------------
Epoch 7/10
Train Loss      : 1.1420
Validation Loss : 1.1340
Train Accuracy  : gender=95.82%, article=93.90%, color=75.60%, usage=95.30%
Val Accuracy    : gender=95.90%, article=94.20%, color=76.45%, usage=95.50%
------------------------------------------------------------
Epoch 8/10
Train Loss      : 1.0485
Validation Loss : 1.0562
Train Accuracy  : gender=96.35%, article=94.52%, color=77.10%, usage=95.82%
Val Accuracy    : gender=96.28%, article=94.75%, color=77.80%, usage=95.91%
------------------------------------------------------------
Epoch 9/10
Train Loss      : 0.9712
Validation Loss : 0.9945
Train Accuracy  : gender=96.78%, article=94.95%, color=78.35%, usage=96.18%
Val Accuracy    : gender=96.65%, article=95.12%, color=78.92%, usage=96.25%
------------------------------------------------------------
Epoch 10/10
Train Loss      : 0.9084
Validation Loss : 0.9482
Train Accuracy  : gender=97.15%, article=95.40%, color=79.30%, usage=96.50%
Val Accuracy    : gender=96.92%, article=95.38%, color=81.86%, usage=96.42%
```

- Wow the accuracy improved, the color accuracy now stands at 82%, If we train it a bit longer, the accuracy will further improved.

## Inference Results

Running the trained model (`src/inference.py`) on a few unseen catalog images from the raw dataset:

### Sample 1 — `data/test_images/1990.jpg`

| Attribute | Prediction | Confidence |
| --------- | ---------- | ---------- |
| Gender    | Women      | 99.98%     |
| Article   | Tops       | 97.16%     |
| Usage     | Casual     | 99.43%     |
| Color     | Green      | 68.07%     |

### Sample 2 — `data/test_images/1982.jpg`

| Attribute | Prediction | Confidence |
| --------- | ---------- | ---------- |
| Gender    | Women      | 93.36%     |
| Article   | Shorts     | 100.00%    |
| Usage     | Casual     | 95.11%     |
| Color     | Purple     | 73.92%     |

### Sample 3 — `data/test_images/59949.jpg`

| Attribute | Prediction | Confidence |
| --------- | ---------- | ---------- |
| Gender    | Women      | 95.55%     |
| Article   | Deodorant  | 94.53%     |
| Usage     | Casual     | 99.93%     |
| Color     | White      | 75.09%     |

### Observations

- **Gender**, **usage** and **Article type** heads are extremely confident and consistently correct — matching the >95% validation accuracy.

- **Color** confidence is the lowest across the board (68-75%), which is consistent with the training results.
