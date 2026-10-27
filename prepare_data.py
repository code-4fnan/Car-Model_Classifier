"""Universal Dataset Splitter for Local Folders."""
import shutil
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw_data"
DATASET_DIR = ROOT / "my_dataset"
VAL_RATIO = 0.2
SEED = 42

def main():
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True)
        print(f"Created {RAW_DIR}. Place your class folders inside it.")
        return

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
        
    class_folders = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if not class_folders:
        print(f"No folders found in {RAW_DIR}. Add image folders before running.")
        return
        
    random.seed(SEED)
    print("Splitting dataset into train and val...")
    
    for cls_path in class_folders:
        cls_name = cls_path.name
        images = [f for f in cls_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        
        if not images:
            continue
            
        random.shuffle(images)
        val_count = max(1, int(len(images) * VAL_RATIO))
        
        val_images = images[:val_count]
        train_images = images[val_count:]
        
        (DATASET_DIR / "train" / cls_name).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "val" / cls_name).mkdir(parents=True, exist_ok=True)
        
        for img in train_images:
            shutil.copy(img, DATASET_DIR / "train" / cls_name / img.name)
        for img in val_images:
            shutil.copy(img, DATASET_DIR / "val" / cls_name / img.name)
            
        print(f"  {cls_name}: {len(train_images)} train | {len(val_images)} val")
        
    print(f"\nDataset successfully prepared in {DATASET_DIR}")

if __name__ == "__main__":
    main()