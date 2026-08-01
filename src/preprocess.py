import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib

important_cols = [
  "gender",
  "articleType",
  "baseColour",
  "usage"
]

usage_map = {
  "Smart Casual": "Casual"
}

COLOR_MAP = {
    "Navy Blue":"Blue",
    "Sky Blue":"Blue",
    "Turquoise Blue":"Blue",

    "Sea Green":"Green",
    "Olive":"Green",

    "Grey Melange":"Grey",

    "Off White":"White",

    "Maroon":"Red"
}

def _fix_bad_line(fields):
  if len(fields) > 10:
    return fields[:9] + [", ".join(fields[9:])]
  if len(fields) < 10:
    return fields + [""] * (10 - len(fields))
  return fields


def load_data(csv_path):
  csv_path = Path(csv_path)
  if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

  df = pd.read_csv(
    csv_path,
    engine="python",
    on_bad_lines=_fix_bad_line,
    encoding="utf-8",
  )
  print(f"Loaded {len(df):,} rows from {csv_path}")
  return df

def drop_missing_value(df):
  before_filtering_data_length = len(df)
  df = df.dropna(subset=important_cols)
  print(f"dropped {before_filtering_data_length - len(df)} rows where data are null")
  return df

def merge_usage(df):
  df["usage"] = df["usage"].replace(usage_map)
  return df

def merge_colors(df):
    df["baseColour"] = df["baseColour"].replace(COLOR_MAP)
    return df

def remove_rare_articles(df, min_samples):
    counts = df["articleType"].value_counts()
    valid_classes = counts[counts >= min_samples].index
    before = len(df)
    df = df[df["articleType"].isin(valid_classes)]
    print(f"Removed {before-len(df)} rows")
    print(f"Remaining article types : {len(valid_classes)}")
    return df

def verify_images(df, image_folder):
    image_folder = Path(image_folder)
    exists = df["id"].apply(
        lambda x: (image_folder / f"{x}.jpg").exists()
    )

    before = len(df)
    df = df[exists]
    print(f"Removed {before-len(df)} rows with missing images")
    return df

def save_dataset(df, output_path):
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(output_path, index=False)
  print(f"Saved cleaned dataset to {output_path}")



def encode_labels(df, encoder_path):
    label_columns = [
        "gender",
        "articleType",
        "baseColour",
        "usage"
    ]

    encoders = {}
    for column in label_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder
        print(f"{column}: {len(encoder.classes_)} classes")
    joblib.dump(encoders, encoder_path)
    print(f"Saved encoders to {encoder_path}")
    print(f"\n{column} mapping:")

    for index, label in enumerate(encoder.classes_):
      print(f"  {index} -> {label}")
    return df

def main():
  project_root = Path(__file__).resolve().parents[1]

  csv_path = project_root / "data" / "raw" / "styles.csv"
  images_path = project_root / "data" / "raw" / "images"
  output_path = project_root / "data" / "processed" / "styles_processed.csv"
  encoder_path = project_root / "data" / "processed" / "label_encoders.pkl"

  df = load_data(csv_path)
  df = drop_missing_value(df)
  df = merge_usage(df)
  df = merge_colors(df)
  df = remove_rare_articles(df, 300)
  df = verify_images(df, images_path)
  df = encode_labels(df, encoder_path)
  save_dataset(df, output_path)


if __name__=="__main__":
    main()