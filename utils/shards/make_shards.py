# Imports
import os
import io

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import INaturalist

import webdataset as wds

# Dataset
dataset_path = '../dataset/species/'

train_dataset = INaturalist(
    root=dataset_path,
    version='2021_train_mini',
    download=False
)

test_dataset = INaturalist(
    root=dataset_path,
    version='2021_valid',
    download=False
)

print(train_dataset)
print("\nTrain sample:")
train_sample = train_dataset[0]
print(train_sample[0].shape)
print()

print(test_dataset)
print("\nTest sample:")
test_sample = test_dataset[0]
print(test_sample[0].shape)
print()

# Shards
shard_dir = os.path.join(dataset_path, "shards")

samples_per_shard = 10000  

def create_shards(dataset, train, out_dir, samples_per_shard: int):
    split = "train" if train else "test"
    
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    writer = wds.ShardWriter(
        os.path.join(split_dir, "data-%06d.tar"),
        maxcount=samples_per_shard
    )
    
    to_pil = T.ToPILImage()
    for idx, (img, label) in enumerate(dataset):
        if not hasattr(img, "save"):
            img = to_pil(img)
            
        key = f"{idx:08d}"
        buffer_io = io.BytesIO()
        img.save(buffer_io, format="JPEG")
        buffer = buffer_io.getvalue()
        
        sample = {
            "__key__": key,
            "jpg": buffer,
            "cls": str(label).encode("utf-8"),
        }
        
        writer.write(sample)
        
    writer.close()

if __name__ == "__main__":
    if not os.path.isdir(shard_dir) or len(os.listdir(shard_dir)) == 0:
        print("Criando shards...")
        
        create_shards(train_dataset, True, shard_dir, samples_per_shard)
        
        create_shards(test_dataset, False, shard_dir, samples_per_shard)