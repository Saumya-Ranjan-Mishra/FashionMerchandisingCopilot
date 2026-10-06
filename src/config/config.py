from pathlib import Path
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_CSV = PROJECT_ROOT / "data" / "processed" / "styles_processed.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"

ENCODER_PATH = PROJECT_ROOT / "data" / "processed" / "label_encoders.pkl"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
INFERENCE_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model_approach_2.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
LOG_INTERVAL = 20
IMAGE_SIZE = 160
BACKBONE = "resnet18"
FREEZE_BACKBONE = False

TRANSFORM = transforms.Compose([
  transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
  )
])