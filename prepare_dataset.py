import os
import shutil
import random
from pathlib import Path

# ==== CONFIG ====
MVT_DIR = "C:/Users/amish/OneDrive/Desktop/Lorayis/mvtec-anomaly detection"  # <- UPDATE THIS
CLASS_NAME = "bottle"                         # <- CHANGE this if needed
DEST_DIR = "./data"                           # Folder inside Lorayis
NUM_TRAIN = 100
NUM_TEST = 20

def copy_subset(src_dir, dst_dir, max_files):
    os.makedirs(dst_dir, exist_ok=True)
    files = list(Path(src_dir).glob("*.png"))
    selected = random.sample(files, min(len(files), max_files))
    for f in selected:
        shutil.copy(f, dst_dir)

def prepare():
    print(f"🔍 Preparing class: {CLASS_NAME}")
    
    class_dir = os.path.join(MVT_DIR, CLASS_NAME)

    train_src = os.path.join(class_dir, "train", "good")
    test_dir = os.path.join(class_dir, "test")

    train_dst = os.path.join(DEST_DIR, "train")
    test_dst = os.path.join(DEST_DIR, "test")

    # Clean old data
    shutil.rmtree(train_dst, ignore_errors=True)
    shutil.rmtree(test_dst, ignore_errors=True)

    # Copy train images (normal only)
    print("📦 Copying training images...")
    copy_subset(train_src, train_dst, NUM_TRAIN)

    # Copy test images (from each test subfolder, e.g. good, broken, etc.)
    print("📦 Copying test images...")
    os.makedirs(test_dst, exist_ok=True)
    test_subfolders = [f.path for f in os.scandir(test_dir) if f.is_dir()]

    for subfolder in test_subfolders:
        copy_subset(subfolder, test_dst, NUM_TEST // len(test_subfolders))

    print("✅ Dataset prepared in ./data")

if __name__ == "__main__":
    prepare()

