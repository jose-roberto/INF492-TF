# Imports
import json
import os

from datetime import datetime
import copy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import ToTensor
from torchvision import transforms, models

from torchinfo import summary

from collections import Counter

# Checking device
print("CUDA?", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print("Device definido:", device)

# Experiment setup
setup = {
    "experiment": "ResNet50_L4U-SGD-CELoss",
    "num_classes": 555,
    "batch_size": 32,
    "num_workers": 8,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "max_epochs": 40,
    "comments": "Sem augmentation, poucas camadas descongeladas"
}

tensorboard_path = './experiments'
models_path = "./models"

os.makedirs(os.path.join(tensorboard_path), exist_ok=True)
os.makedirs(os.path.join(models_path), exist_ok=True)

# Tensorboard functions
def plot_net_attributes(epoch, net, writer):
    layers = list(net.modules())

    layer_id = 1
    for layer in layers:
        if isinstance(layer, nn.Linear) :
            writer.add_histogram(f'Bias/linear-{layer_id}', layer.bias, epoch )
            writer.add_histogram(f'Weight/linear-{layer_id}', layer.weight, epoch )
            writer.add_histogram(f'Grad/linear-{layer_id}', layer.weight.grad, epoch )
            layer_id += 1

# Transforms
mean = [0.493, 0.509, 0.463]
std = [0.223, 0.222, 0.268]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
                             
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomRotation(degrees=15),

    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    # transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),
                             
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),

    # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# NABirds
## Dataset
class NABirdsDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, remap_labels=True):

        self.root = root_dir
        self.transform = transform

        self.id2path = {}
        with open(os.path.join(root_dir, "images.txt"), "r") as f:
            for line in f:
                parts = line.strip().split()
                
                if len(parts)!=2: continue
                    
                img_id, rel = parts
                self.id2path[img_id] = rel

        raw_id2label = {}
        with open(os.path.join(root_dir, "image_class_labels.txt"), "r") as f:
            for line in f:
                parts = line.strip().split()
                
                if len(parts)!=2: continue
                    
                img_id, cls = parts
                raw_id2label[img_id] = int(cls)

        flag_target = '1' if split=='train' else '0'
        raw_samples = []
        with open(os.path.join(root_dir, "train_test_split.txt"), "r") as f:
            for line in f:
                parts = line.strip().split()
                
                if len(parts)!=2: continue
                    
                img_id, flag = parts
                if flag==flag_target and img_id in self.id2path:
                    raw_samples.append((img_id, raw_id2label[img_id]))

        if remap_labels:
            unique_labels = sorted({lbl for _,lbl in raw_samples})

            self.label_map = {old: new for new, old in enumerate(unique_labels)}

            self.samples = [(img_id, self.label_map[lbl]) for img_id, lbl in raw_samples]
            self.num_classes = len(unique_labels)
        else:
            self.samples = raw_samples
            self.num_classes = len(sorted({lbl for _,lbl in raw_samples})) 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        img = Image.open(os.path.join(self.root, "images", self.id2path[img_id])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

data_root = "../dataset/nabirds/versions/1"

train_dataset = NABirdsDataset(
    data_root,
    split='train',
    transform=train_transform
)

val_dataset = NABirdsDataset(
    data_root,
    split='val',
    transform=val_transform
)

## Dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=setup["batch_size"],
    shuffle=True,
    num_workers=setup["num_workers"],
    pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=setup["batch_size"],
    shuffle=False,
    num_workers=setup["num_workers"],
    pin_memory=True
)

# Architecture
weights = models.ResNet50_Weights.IMAGENET1K_V1

net = models.resnet50(weights=weights)

for param in net.parameters():
    param.requires_grad = False

for name, param in net.layer4.named_parameters():
        param.requires_grad = True

in_features = net.fc.in_features

net.fc = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features, setup["num_classes"])
)

# Criterion class weights
def calculate_class_weights():
    counts = Counter([lbl for _,lbl in train_dataset.samples])
    total = sum(counts.values())
    
    class_weights = [ total / counts[i] if counts[i]>0 else 0.0
                      for i in range(setup["num_classes"]) ]
    
    s = sum(class_weights)
    class_weights = [w/s for w in class_weights]

    return class_weights
    
# Train
def train(net, train_dataloader, val_dataloader, device, tensorboard_path):
  
    net.to(device)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=setup['lr'],
        weight_decay=setup['weight_decay'],
        momentum=setup['momentum'])
    
    # class_weights = calculate_class_weights()
    # weight_tensor = tensor(class_weights, dtype=torch.float32).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    criterion = nn.CrossEntropyLoss()

    criterion.to(device)

    now = datetime.now()
    tensorboard_path = os.path.join(tensorboard_path, setup['experiment'] + '-' + now.strftime("%Y%m%d_%H%M%S"))
    
    writer = SummaryWriter(log_dir = tensorboard_path)
    writer.add_graph(net, next(iter(train_dataloader))[0].to(device))

    setup_path = os.path.join(tensorboard_path, 'setup.txt')
    with open(setup_path, 'w', encoding='utf-8') as f:
        json.dump(setup, f, indent=4, ensure_ascii=False, default=str)

    max_accuracy = -1.0

    for epoch in tqdm(range(setup['max_epochs'])):
        
        net.train()  

        train_loss, train_accuracy = [], []

        for train_batch in train_dataloader:
            
            train_x, train_label = train_batch
            train_x = train_x.to(device)
            train_label = train_label.to(device)

            optimizer.zero_grad()

            outs = net(train_x)
    
            loss = criterion(outs, train_label)

            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()

            predict_labels = torch.max(outs, axis=1)[1]
            correct = torch.sum(predict_labels == train_label).item()
            accuracy = correct/train_label.size(0)

            train_accuracy.append(accuracy)

            plot_net_attributes(epoch, net, writer)

        train_loss = np.asarray(train_loss)
        train_accuracy = np.asarray(train_accuracy)

        val_loss, val_accuracy = validate(net, criterion, val_dataloader, device)

        writer.add_scalar('Loss/train', train_loss.mean(), epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy.mean()*100., epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            
        if val_accuracy > max_accuracy:
            best_model = copy.deepcopy(net)
            max_accuracy = val_accuracy
            print(f'Salvando o modelo com melhor acurácia: {max_accuracy:3.2f}%')
            
        print(f'Época: {epoch+1:3d} | Loss/train: {train_loss.mean():3.4f}% | Accuracy/train: {train_accuracy.mean()*100.:3.4f}% |\
            Loss/val: {val_loss:3.4f}% | Accuracy/val: {val_accuracy:3.4f}% |')

    path = f'{models_path}/{setup["experiment"]}-{max_accuracy:.2f}.pkl'
    torch.save(best_model, path)
    print(f'Melhor modelo salvo em: {path}')

    writer.flush()
    writer.close()
    
    return best_model
    
# Validate
def validate(net, criterion, val_dataloader, device):

    net.eval()
    net.to(device)

    val_loss, val_accuracy = [], []

    for test_batch in val_dataloader:

        test_x, test_label = test_batch
        test_x = test_x.to(device)
        test_label = test_label.to(device)

        with torch.no_grad():
            outs = net(test_x).detach()

            loss = criterion(outs, test_label)
            
            val_loss.append(loss.item())
    
            predict_labels = torch.max(outs, axis=1)[1]
            correct = torch.sum(predict_labels == test_label).item()
            accuracy = correct/test_label.size(0)
    
            val_accuracy.append(accuracy)
        
    val_loss = np.asarray(val_loss)
    val_accuracy = np.asarray(val_accuracy)

    return val_loss.mean(), val_accuracy.mean()*100.

# Fit
best_model = train(net, train_dataloader, val_dataloader, device, tensorboard_path)
