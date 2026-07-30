import os

import pandas as pd
import torch

from torch.utils.data import Dataset
from PIL import Image


class FashionDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):

        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        image_id = row["id"]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        gender = torch.tensor(row["gender"], dtype=torch.long)
        article_type = torch.tensor(row["articleType"], dtype=torch.long)
        color = torch.tensor(row["baseColour"], dtype=torch.long)
        usage = torch.tensor(row["usage"], dtype=torch.long)

        return (image, gender, article_type, color, usage)