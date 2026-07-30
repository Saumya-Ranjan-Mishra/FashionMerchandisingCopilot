from Dataset.FashionDataset import FashionDataset
from config import config
import torch
from torch.utils.data import DataLoader, random_split
import joblib
import multiprocessing
from models import models
from models.model_trainer import Trainer

def main():
    # read the data
    fashion_dataset = FashionDataset(csv_file=config.DATA_CSV, image_dir=config.IMAGE_DIR, transform=config.TRANSFORM)

    # get total data size
    dataset_size = len(fashion_dataset)
    print(f"[info] Loaded dataset with {dataset_size} samples")

    # get the devide the datasize of train, val & test set to 70%, 20% and 10%
    train_size = int(dataset_size * 0.7)
    val_size = int(dataset_size * 0.2)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)

    # split the complete dataset randomly (with manual_seed) into train, test and val set
    train_dataset, val_dataset, test_dataset = random_split(fashion_dataset, [train_size, val_size, test_size], generator=generator)
    print(f"[info] Split sizes -> train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    pin_memory = config.DEVICE == "cuda"

    # create the dataloader from the dataset. it acts as an Iterator, where It will call the __getitem__ method and will return the batch of 32 data on each iteration.
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    _test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    print(
        f"[info] DataLoader setup -> batch_size={config.BATCH_SIZE}, "
        f"num_workers={config.NUM_WORKERS}, pin_memory={pin_memory}"
    )

    # get the number of classes for each category so that we can get the number of output neurons we need for each calss and create the classifier layer
    encoders = joblib.load(config.ENCODER_PATH)

    num_gender_classes = len(encoders["gender"].classes_)
    num_article_classes = len(encoders["articleType"].classes_)
    num_color_classes = len(encoders["baseColour"].classes_)
    num_usage_classes = len(encoders["usage"].classes_)
    print(
        "[info] Class counts -> "
        f"gender={num_gender_classes}, article={num_article_classes}, "
        f"color={num_color_classes}, usage={num_usage_classes}"
    )

    # create the model instance
    model = models.MultiTaskFashionModel(
        num_gender_classes,
        num_article_classes,
        num_color_classes,
        num_usage_classes,
        freeze_backbone=config.FREEZE_BACKBONE
    )
    print(f"[info] Backbone -> {config.BACKBONE}, freeze_backbone={config.FREEZE_BACKBONE}")

    device = torch.device(config.DEVICE)

    model.to(device)
    print(f"[info] Training on device: {device}")

    # train the model
    trainer = Trainer(
        model=model,
        train_loader=train_data_loader,
        val_loader=val_data_loader,
        device=device,
        learning_rate=config.LEARNING_RATE,
        checkpoint_path=config.CHECKPOINT_PATH,
        log_interval=config.LOG_INTERVAL,
    )
    print("[info] Starting training loop...")
    trainer.train(num_epochs=config.NUM_EPOCHS)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()