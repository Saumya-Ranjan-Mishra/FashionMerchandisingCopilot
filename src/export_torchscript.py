import argparse
import json
from pathlib import Path

import joblib
import torch
import torch.nn as nn

from config import config
from models.models import MultiTaskFashionModel
from preprocess import COLOR_MAP


OUTPUT_ORDER = ("gender", "article", "usage", "color")


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images)
        return tuple(outputs[name] for name in OUTPUT_ORDER)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the split-head fashion model to TorchScript."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.PROJECT_ROOT / "checkpoints" / "best_model_approach_3.pth",
        help="Approach 3 state-dict checkpoint (defaults to the split-head model).",
    )
    parser.add_argument(
        "--encoders",
        type=Path,
        default=(
            config.ENCODER_PATH
            if config.ENCODER_PATH.is_file()
            else config.PROJECT_ROOT / "encoders" / "label_encoders.pkl"
        ),
        help="Joblib file containing the fitted label encoders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.PROJECT_ROOT / "checkpoints" / "fashion_model.pt",
        help="Destination TorchScript file.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, dict) or not any(
        key.startswith("initial_layers.") for key in state_dict
    ):
        raise ValueError(
            f"{checkpoint_path.name} is not the split-head Approach 3 checkpoint. "
            "Use checkpoints/best_model_approach_3.pth; Approaches 1 and 2 "
            "use the older single-trunk architecture."
        )
    return state_dict


def load_class_counts(state_dict):
    try:
        return {
            "gender": int(state_dict["gender_head.5.weight"].shape[0]),
            "article": int(state_dict["article_head.5.weight"].shape[0]),
            "color": int(state_dict["color_head.5.weight"].shape[0]),
            "usage": int(state_dict["usage_head.5.weight"].shape[0]),
        }
    except KeyError as error:
        raise ValueError(
            "Checkpoint does not contain the expected split-head classifier weights."
        ) from error


def load_class_labels(encoder_path):
    if not encoder_path.is_file():
        raise FileNotFoundError(f"Label encoders not found: {encoder_path}")
    encoders = joblib.load(encoder_path)
    encoder_keys = {
        "gender": "gender",
        "article": "articleType",
        "color": "baseColour",
        "usage": "usage",
    }
    classes_by_name = {
        name: [str(label) for label in encoders[encoder_key].classes_]
        for name, encoder_key in encoder_keys.items()
    }
    classes_by_name["color"] = sorted(
        {COLOR_MAP.get(label, label) for label in classes_by_name["color"]}
    )
    return classes_by_name


def build_output_metadata(classes_by_name, class_counts):
    metadata = {}
    for index, name in enumerate(OUTPUT_ORDER):
        labels = classes_by_name[name]
        mapping_matches = len(labels) == class_counts[name]
        if not mapping_matches:
            mapping_status = "unavailable_class_count_mismatch"
            print(
                f"WARNING: {name} checkpoint has {class_counts[name]} classes, "
                f"but the encoder has {len(labels)}. Its label mapping is omitted."
            )
        elif name == "color":
            mapping_status = "class_count_matches_training_preprocessing"
        else:
            mapping_status = "class_count_matches_available_encoder"

        metadata[name] = {
            "tuple_index": index,
            "class_count": class_counts[name],
            "classes": labels if mapping_matches else None,
            "label_mapping_status": mapping_status,
        }
    return metadata


def verify_torchscript(export_model, traced_model):
    for batch_size in (1, 2):
        images = torch.zeros(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        expected = export_model(images)
        actual = traced_model(images)
        for expected_output, actual_output in zip(expected, actual):
            torch.testing.assert_close(actual_output, expected_output)


def main():
    args = parse_args()
    state_dict = load_checkpoint(args.checkpoint)
    class_counts = load_class_counts(state_dict)
    classes_by_name = load_class_labels(args.encoders)

    model = MultiTaskFashionModel(
        num_gender_classes=class_counts["gender"],
        num_article_classes=class_counts["article"],
        num_color_classes=class_counts["color"],
        num_usage_classes=class_counts["usage"],
        freeze_backbone=False,
        pretrained_backbone=False,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    export_model = ExportWrapper(model).eval()
    example = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    traced_model = torch.jit.trace(export_model, example, strict=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(args.output))

    loaded_model = torch.jit.load(str(args.output), map_location="cpu").eval()
    with torch.inference_mode():
        verify_torchscript(export_model, loaded_model)

    metadata = {
        "input": {
            "shape": ["batch", 3, config.IMAGE_SIZE, config.IMAGE_SIZE],
            "color_order": "RGB",
            "layout": "NCHW",
            "dtype": "float32",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "outputs": build_output_metadata(classes_by_name, class_counts),
        "output_format": "logits; apply softmax along the class dimension",
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported and verified TorchScript model: {args.output}")
    print(f"Wrote input/output metadata and class labels: {metadata_path}")
    print(f"Output tuple order: {', '.join(OUTPUT_ORDER)}")


if __name__ == "__main__":
    main()