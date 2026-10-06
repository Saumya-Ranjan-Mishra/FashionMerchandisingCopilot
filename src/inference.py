import json
from pathlib import Path
import joblib
from PIL import Image
import torch
import torchvision.transforms as transforms

from config import config
from models import legacy_model


class FashionInferenceEngine:

    def __init__(self, checkpoint_path, encoder_path, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 1. Load Label Encoders
        if not Path(encoder_path).exists():
            raise FileNotFoundError(f"Encoder file not found at {encoder_path}")

        self.encoders = joblib.load(encoder_path)

        # Map task names to encoder keys
        self.task_encoder_map = {
            "gender": "gender",
            "article": "articleType",
            "color": "baseColour",
            "usage": "usage",
        }

        num_gender_classes = len(self.encoders["gender"].classes_)
        num_article_classes = len(self.encoders["articleType"].classes_)
        num_color_classes = len(self.encoders["baseColour"].classes_)
        num_usage_classes = len(self.encoders["usage"].classes_)

        self.model = legacy_model.LegacyResNetFashionModel(
            num_gender_classes=num_gender_classes,
            num_article_classes=num_article_classes,
            num_color_classes=num_color_classes,
            num_usage_classes=num_usage_classes,
            freeze_backbone=True,
        )

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint file not found at {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image_input):
        """Predicts attributes for a given image path or PIL Image object.

        Returns a structured dictionary with predicted labels and confidence
        scores.
        """
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found at {image_path}")
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(
                "Input must be a file path string or PIL Image object."
            )

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        predictions = {}

        # Decode Predictions
        for task_name, output in outputs.items():
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

            idx = predicted_idx.item()
            conf_score = confidence.item()

            # Decode class index back to original label string
            encoder_key = self.task_encoder_map[task_name]
            label_encoder = self.encoders[encoder_key]
            predicted_label = label_encoder.inverse_transform([idx])[0]

            predictions[task_name] = {
                "label": predicted_label,
                "confidence": round(conf_score, 4),
            }

        return predictions


# ==========================================
# Example Usage Execution
# ==========================================
if __name__ == "__main__":
    # Checkpoint and Encoder paths
    checkpoint_file = config.INFERENCE_CHECKPOINT_PATH
    encoder_file = config.ENCODER_PATH

    # Initialize Engine
    engine = FashionInferenceEngine(
        checkpoint_path=checkpoint_file, encoder_path=encoder_file
    )

    # Raw image path
    sample_image_path = Path(
        r".\images\59949.jpg"
    )

    if sample_image_path.exists():
        print(f"Running inference on: {sample_image_path}\n")
        results = engine.predict(sample_image_path)

        # Print Clean Structured Output
        print("--- Model Predictions ---")
        for task, data in results.items():
            print(
                f"{task.capitalize():<10}: {data['label']:<20} (Confidence: {data['confidence']:.2%})"
            )

        # Extract clean JSON payload for LLM prompt generation
        llm_payload = {task: data["label"] for task, data in results.items()}
        print("\n--- Clean JSON for LLM Prompting ---")
        print(json.dumps(llm_payload, indent=2))
    else:
        print(f"Image not found at {sample_image_path}")