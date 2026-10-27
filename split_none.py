import os
import random
import shutil
from pathlib import Path

source_dir = Path("random_images") 
train_dir = Path("my_dataset/train/None")
val_dir = Path("my_dataset/val/None")

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

images = [f for f in source_dir.rglob("*.*") if f.is_file()]
random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

for img in train_images:
    shutil.copy(img, train_dir / img.name)
for img in val_images:
    shutil.copy(img, val_dir / img.name)

print(f"Copied {len(train_images)} to train/None and {len(val_images)} to val/None.")