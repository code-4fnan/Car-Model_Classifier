"""Train the transfer learning model on vehicle body types."""
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import create_model

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "my_dataset"
CHECKPOINT_PATH = ROOT / "resnet18_cars.pth"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Add data augmentation to the training transform pipeline
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATASET_DIR / "val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Training on {len(train_dataset.classes)} classes.")
    
    # Load model dynamically based on folder count
    model, base_transform = create_model(num_classes=len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Optimize only the parameters that require gradients (the new FC layer)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Acc: {100.0 * train_correct / train_total:.2f}% | "
            f"Val Acc: {100.0 * val_correct / val_total:.2f}%"
        )

    # Save Checkpoint
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "class_to_idx": train_dataset.class_to_idx,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Model saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()