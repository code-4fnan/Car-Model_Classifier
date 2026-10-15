"""Train the custom three-block CNN for 10 epochs."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CarCNN

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "my_dataset"
CHECKPOINT_PATH = ROOT / "car_cnn_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if not (DATASET_DIR / "train").is_dir() or not (DATASET_DIR / "val").is_dir():
        raise SystemExit(
            "Dataset folders are missing. Run `python prepare_data.py` first."
        )

    # Mean/std 0.5 maps each channel approximately to [-1, 1].
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.ImageFolder(
        root=str(DATASET_DIR / "train"), transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root=str(DATASET_DIR / "val"), transform=transform
    )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise SystemExit(
            "Train and validation class mappings differ. Ensure both splits "
            "contain the same class folders."
        )
    if len(train_dataset.classes) != 2:
        raise SystemExit(
            f"Expected exactly 2 classes, found {train_dataset.classes}"
        )
    for class_name in train_dataset.classes:
        train_count = sum(1 for _, label in train_dataset.samples
                          if train_dataset.classes[label] == class_name)
        val_count = sum(1 for _, label in val_dataset.samples
                        if val_dataset.classes[label] == class_name)
        if train_count == 0 or val_count == 0:
            raise SystemExit(
                f"Class '{class_name}' must have images in train and val."
            )

    # num_workers=0 is reliable for Windows and avoids multiprocessing issues.
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {train_dataset.class_to_idx}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Epochs: {EPOCHS}; batch size: {BATCH_SIZE}; learning rate: {LEARNING_RATE}")

    model = CarCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += batch_size

        train_loss = train_loss_sum / max(train_total, 1)
        train_accuracy = 100.0 * train_correct / max(train_total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += batch_size

        val_loss = val_loss_sum / max(val_total, 1)
        val_accuracy = 100.0 * val_correct / max(val_total, 1)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.2f}% "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.2f}%"
        )

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "class_to_idx": train_dataset.class_to_idx,
        "classes": train_dataset.classes,
        "image_size": IMAGE_SIZE,
        "normalization_mean": [0.5, 0.5, 0.5],
        "normalization_std": [0.5, 0.5, 0.5],
        "architecture": "CarCNN",
        "epochs": EPOCHS,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"\nSaved model checkpoint: {CHECKPOINT_PATH}")
    print("Training finished. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
