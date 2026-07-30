import torch
import torch.nn as nn
from pathlib import Path


class Trainer:

    def __init__(self, model, train_loader, val_loader, device, learning_rate, checkpoint_path="best_model.pth", log_interval=200):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.checkpoint_path = checkpoint_path
        self.log_interval = log_interval

        self.optimizer = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad,
                self.model.parameters()
            ),
            lr=learning_rate,
            weight_decay=1e-8
        )

        self.best_validation_loss = float("inf")

    def _init_accuracy_tracker(self):
        return {
            "gender": {"correct": 0, "total": 0},
            "article": {"correct": 0, "total": 0},
            "color": {"correct": 0, "total": 0},
            "usage": {"correct": 0, "total": 0},
        }

    def _update_accuracy_tracker(self, tracker, outputs, targets):
        for task_name, target in targets.items():
            pred = outputs[task_name].argmax(dim=1)
            tracker[task_name]["correct"] += (pred == target).sum().item()
            tracker[task_name]["total"] += target.size(0)

    def _compute_accuracy(self, tracker):
        accuracy = {}
        for task_name, stats in tracker.items():
            if stats["total"] == 0:
                accuracy[task_name] = 0.0
            else:
                accuracy[task_name] = stats["correct"] / stats["total"]
        return accuracy

    def train(self, num_epochs):

        for epoch in range(num_epochs):
            print(f"[epoch {epoch+1}/{num_epochs}] training...")
            train_loss, train_acc = self.train_one_epoch()
            print(f"[epoch {epoch+1}/{num_epochs}] validating...")
            validation_loss, val_acc = self.validate()

            print("-" * 60)
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss      : {train_loss:.4f}")
            print(f"Validation Loss : {validation_loss:.4f}")
            print(
                "Train Accuracy  : "
                f"gender={train_acc['gender']:.2%}, "
                f"article={train_acc['article']:.2%}, "
                f"color={train_acc['color']:.2%}, "
                f"usage={train_acc['usage']:.2%}"
            )
            print(
                "Val Accuracy    : "
                f"gender={val_acc['gender']:.2%}, "
                f"article={val_acc['article']:.2%}, "
                f"color={val_acc['color']:.2%}, "
                f"usage={val_acc['usage']:.2%}"
            )

            print("-" * 60)

            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_path
                )

                print("Best Model Saved")

    def train_one_epoch(self):

        self.model.train()
        running_loss = 0
        accuracy_tracker = self._init_accuracy_tracker()

        for step, (images, gender, article, color, usage) in enumerate(self.train_loader, start=1):
            images = images.to(self.device)
            gender = gender.to(self.device)
            article = article.to(self.device)
            color = color.to(self.device)
            usage = usage.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            gender_loss = self.loss_fn(outputs["gender"], gender)
            article_loss = self.loss_fn(outputs["article"], article)
            color_loss = self.loss_fn(outputs["color"], color)
            usage_loss = self.loss_fn(outputs["usage"], usage)

            targets = {
                "gender": gender,
                "article": article,
                "color": color,
                "usage": usage,
            }
            self._update_accuracy_tracker(accuracy_tracker, outputs, targets)

            total_loss = (gender_loss + article_loss + 5*color_loss + usage_loss)
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()

            if self.log_interval > 0 and step % self.log_interval == 0:
                avg_loss_so_far = running_loss / step
                print(
                    f"  [train] step {step}/{len(self.train_loader)} "
                    f"avg_loss={avg_loss_so_far:.4f}"
                )

        return running_loss / len(self.train_loader), self._compute_accuracy(accuracy_tracker)

    def validate(self):

        self.model.eval()
        running_loss = 0
        accuracy_tracker = self._init_accuracy_tracker()

        with torch.no_grad():
            for step, (images, gender, article, color, usage) in enumerate(self.val_loader, start=1):

                images = images.to(self.device)
                gender = gender.to(self.device)
                article = article.to(self.device)
                color = color.to(self.device)
                usage = usage.to(self.device)

                outputs = self.model(images)
                gender_loss = self.loss_fn(outputs["gender"], gender)
                article_loss = self.loss_fn(outputs["article"], article)
                color_loss = self.loss_fn(outputs["color"], color)
                usage_loss = self.loss_fn(outputs["usage"], usage)

                targets = {
                    "gender": gender,
                    "article": article,
                    "color": color,
                    "usage": usage,
                }
                self._update_accuracy_tracker(accuracy_tracker, outputs, targets)

                total_loss = (gender_loss + article_loss + color_loss + usage_loss)
                running_loss += total_loss.item()

                if self.log_interval > 0 and step % self.log_interval == 0:
                    avg_val_loss_so_far = running_loss / step
                    print(
                        f"  [val] step {step}/{len(self.val_loader)} "
                        f"avg_loss={avg_val_loss_so_far:.4f}"
                    )

        return running_loss / len(self.val_loader), self._compute_accuracy(accuracy_tracker)