"""
Dataset Preparation: Consolidate Stanford Cars into General Body Types.
"""
import random
import shutil
import scipy.io as sio
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw_data"
DATASET_DIR = ROOT / "my_dataset"
VAL_RATIO = 0.2
SEED = 42

def determine_body_type(class_name: str) -> str:
    """Map Stanford Car specific class names to general body types."""
    name = class_name.lower()
    if "suv" in name: return "SUV"
    if "sedan" in name: return "Sedan"
    if "coupe" in name: return "Coupe"
    if "convertible" in name or "roadster" in name: return "Convertible"
    if "hatchback" in name: return "Hatchback"
    if "cab" in name: return "Pickup_Truck"
    if "van" in name or "minivan" in name: return "Van"
    if "wagon" in name: return "Wagon"
    
    # Fallback for sports cars / exotics that lack standard body type text
    sports_brands = [
        "ferrari", "lamborghini", "bugatti", "mclaren", 
        "corvette", "aston martin", "ford gt", "audi r8", "srt-8"
    ]
    if any(brand in name for brand in sports_brands):
        return "Sports_Car"
    
    return "Other"

def main():
    # Auto-detect folder structure (nested vs flattened)
    if (RAW_DIR / "cars_train" / "cars_train").exists():
        train_images_dir = RAW_DIR / "cars_train" / "cars_train"
    else:
        train_images_dir = RAW_DIR / "cars_train"
        
    if (RAW_DIR / "car_devkit" / "devkit" / "cars_train_annos.mat").exists():
        annotations_path = RAW_DIR / "car_devkit" / "devkit" / "cars_train_annos.mat"
        meta_path = RAW_DIR / "car_devkit" / "devkit" / "cars_meta.mat"
    else:
        annotations_path = RAW_DIR / "car_devkit" / "cars_train_annos.mat"
        meta_path = RAW_DIR / "car_devkit" / "cars_meta.mat"

    if not train_images_dir.exists() or not annotations_path.exists():
        raise FileNotFoundError(
            f"\nRaw data not found.\n"
            f"Expected images at: {train_images_dir}\n"
            f"Expected annotations at: {annotations_path}\n"
            f"Please ensure you placed 'cars_train' and 'car_devkit' directly inside 'raw_data'."
        )

    # Load class names
    meta = sio.loadmat(str(meta_path))
    class_names = [c[0] for c in meta['class_names'][0]]
    
    # Create mapping from Class ID (1-196) to Body Type
    id_to_body = {}
    for i, name in enumerate(class_names, start=1):
        id_to_body[i] = determine_body_type(name)

    # Load image annotations
    annos = sio.loadmat(str(annotations_path))
    annotations = annos['annotations'][0]

    # Group images by Body Type
    body_to_images = {body: [] for body in set(id_to_body.values())}
    for anno in annotations:
        img_name = anno['fname'][0]
        class_idx = int(anno['class'][0][0])
        body_type = id_to_body[class_idx]
        body_to_images[body_type].append(img_name)

    # Reset and create dataset directories
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    random.seed(SEED)
    
    print("Processing and splitting images into body types...")
    for body_type, images in body_to_images.items():
        if body_type == "Other":
            continue  # Skip cars we couldn't cleanly classify
            
        (train_dir / body_type).mkdir(parents=True, exist_ok=True)
        (val_dir / body_type).mkdir(parents=True, exist_ok=True)

        random.shuffle(images)
        val_count = max(1, int(len(images) * VAL_RATIO))
        
        val_images = images[:val_count]
        train_images = images[val_count:]

        for img in train_images:
            shutil.copy(train_images_dir / img, train_dir / body_type / img)
        for img in val_images:
            shutil.copy(train_images_dir / img, val_dir / body_type / img)
            
        print(f"  {body_type}: {len(train_images)} train | {len(val_images)} val")

    print(f"\nSuccessfully organized body types into {DATASET_DIR}")

if __name__ == "__main__":
    main()