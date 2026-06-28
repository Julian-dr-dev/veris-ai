

#training script

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np



from models.ijepa import (
    build_ijepa_small,
    build_ijepa_base,
    create_masks

)

class Config:
    train_dir = "data/mvtec/bottle/train/good"
    val_split = 0.1
    img_size = 224
    num_workers = 2

    variant    = "small"
    patch_size = 16
    ema_decay  = 0.996

    num_target_blocks = 4
    target_scale      = (0.15, 0.2)
    target_ratio      = (0.75, 1.5)
    context_scale     = 0.85

    epochs        = 100
    batch_size    = 16
    lr            = 1e-4
    min_lr        = 1e-6
    weight_decay  = 0.04
    warmup_epochs = 10
 
    # Checkpointing
    checkpoint_dir = "data/checkpoints"
    save_every     = 10
 
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()




#Dataset: 
class NormalImageDataset(Dataset):

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(self, image_dir: str, img_size: int, augment: bool = True):
        self.img_size = img_size
        self.augment = augment

        self.paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if os.path.splittext(f)[1].lower() if self.EXTENSIONS

        ]

        if not self.paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        
        print("  [dataset] Found {len(self.paths)} images in {image_dir}")

        self.train_transform = transforms.Compose([
            transforms.Reize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),

            transforms.ColorJitter(
                brightness=0.2,
                contraist=0.2,
                saturation=0.2,
                hue=0.05,
            ),

            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],

            ),

        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.augment:
            return self.train_transform(img)
        else:
            return self.val_transform(img)



        
def build_dataloaders(cfg):

    full_dataset = NormalImageDataset(
        cfg.train_dir, cfg,img_size, augment=False
    )
    
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * cfg.val_split))
    n_train = n_total - n_val

    train_paths = full_dataset.paths[:n_train]
    val_paths = full_dataset.paths[n_train:]

    print(f"  [data] Train: {len(train_paths)} | Val: {len(val_paths)}")

    class SubsetDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)
        

    train_transform = NormalImageDataset(
        cfg.train_dir, cfg.img_size, augment=True
    ).train_transform

    val_transform = NormalImageDataset(
        cfg.train_dir, cfg.img_size, augment=False
    ).val_transform


    train_ds = SubsetDataset(train_paths, train_transform)
    val_ds = SubsetDataset(val_paths, val_transform)

    train_loader = DataLoader(
        

    )








    

    










