import os
from PIL import Image

dir_paths = {
    "train": "../dataset/species/2021_train_mini",
    "valid": "../dataset/species/2021_valid"
}

target_size = (384, 384)
jpeg_quality = 90 

def preprocess_split(src_root):
    for root, _, files in os.walk(src_root):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
                
            src_path = os.path.join(root, fname)
            
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                resized = img.resize(target_size, Image.LANCZOS)
                resized.save(src_path, format="JPEG", quality=jpeg_quality)

if __name__ == "__main__":
    for split in (dir_paths):
        print(f"Resize no split: {split}")
        preprocess_split(dir_paths[split])