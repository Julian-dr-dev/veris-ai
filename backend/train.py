

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
            if os.path.splitext(f)[1].lower() if self.EXTENSIONS

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
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader



#Learning rate scheduler: 
def get_lr(epoch: int, cfg) -> float:

        if epoch < cfg.warmup_epochs: 
            return cfg.lr * (epoch + 1) / cfg.warmup_epochs
        else:
            progress = (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs)
            cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
            return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine
        



#train an epoch:

def train_epoch(model, loader, optimizer, cfg):
    model.train()
    total_loss = 0.0
    num_patches = model.patch_grid ** 2

    for imgs in tqdm(loader, desc=" train", leave=False):
        imgs = imgs.to(cfg.device)

        context_indices, target_blocks = create_masks(
            num_patches,
            num_target_blocks=cfg.num_target_blocks,
            target_scale=cfg.target_scale,
            target_ratio=cfg.target_ratio,
            context_scale=cfg.context_scale,
            patch_grid=model.patch_grid,
        )
        context_indices = context_indices.to(cfg.device)

        loss = model(imgs, context_indices, target_blocks)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        model.update_target_encoder()
        total_loss += loss.item()

    
    return total_loss / len(loader)

def val_epcoh(model, loader, cfg):
    model.eval()
    total_loss = 0.0
    num_patches = model.patch_grid ** 2

    with torch.no_grad():
        for imgs in tqdm(loader, desc=" val ", leave=False):
            imgs = imgs.to(cfg.device)

            context_indices, target_blocks = create_masks(
                num_patches,
                num_target_blocks=cfg.num_target_blocks,
                target_scale=cfg.target_scale,
                target_ratio=cfg.target_ratio,
                context_scale=cfg.context_scale,
                patch_grid=model.patch_grid,
            )

            context_indices = context_indices.to(cfg.device)
 
            loss = model(imgs, context_indices, target_blocks)
            total_loss += loss.item()

    return total_loss / len(loader)




#checkpoint:
def save_checkpoint(model, optimizer, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": {
            "variant":    cfg.variant,
            "img_size":   cfg.img_size,
            "patch_size": cfg.patch_size,
            "ema_decay":  cfg.ema_decay,
        }
    }, path)

    print(f"    [ckpt] Saved ->{path}")


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [ckpt] Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"]




#plot curves: 
def plot_curves(train_losses, val_losses, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="train loss", color="steelblue")
    ax.plot(val_losses,   label="val loss",   color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("I-JEPA Loss")
    ax.set_title("Training Curves — Veris AI (I-JEPA)")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [plot] Saved → {path}")
    











#main:
def main():
    print(f"\n{'='*56}")
    print(f"  Veris AI — I-JEPA Anomaly Detection Training")
    print(f"  Dataset:  MVTec AD — Bottle (normal only)")
    print(f"  Variant:  ViT-{cfg.variant.capitalize()}")
    print(f"  Device:   {cfg.device}")
    print(f"  Epochs:   {cfg.epochs} | Batch: {cfg.batch_size}")
    print(f"{'='*56}\n")
 
    # Data
    print("Loading data...")
    train_loader, val_loader = build_dataloaders(cfg)
 
    # Model
    print("Building model...")
    if cfg.variant == "small":
        model = build_ijepa_small(img_size=cfg.img_size)
    else:
        model = build_ijepa_base(img_size=cfg.img_size)
 
    model = model.to(cfg.device)
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
 
    # Optimizer — only parameters with requires_grad=True
    # Target encoder is excluded automatically since its
    # requires_grad=False was set in IJEPA.__init__
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
 
    # Resume if checkpoint exists
    start_epoch = 0
    latest_ckpt = os.path.join(cfg.checkpoint_dir, "ijepa_latest.pth")
    if os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}...")
        start_epoch = load_checkpoint(latest_ckpt, model, optimizer) + 1
 
    # Training loop
    train_losses  = []
    val_losses    = []
    best_val_loss = float("inf")
 
    print(f"\nStarting training from epoch {start_epoch + 1}/{cfg.epochs}\n")
 
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
 
        # Set learning rate for this epoch
        lr = get_lr(epoch, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
 
        train_loss = train_epoch(model, train_loader, optimizer, cfg)
        val_loss   = val_epoch(model, val_loader, cfg)
 
        elapsed = time.time() - t0
 
        print(
            f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={lr:.2e}  t={elapsed:.1f}s"
        )
 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
 
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(cfg.checkpoint_dir, "ijepa_best.pth")
            )
 
        # Save periodic
        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(cfg.checkpoint_dir, f"ijepa_epoch{epoch+1}.pth")
            )
 
        # Save latest for resuming
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            os.path.join(cfg.checkpoint_dir, "ijepa_latest.pth")
        )
 
    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {cfg.checkpoint_dir}/ijepa_best.pth")
 
    plot_curves(
        train_losses,
        val_losses,
        os.path.join(cfg.checkpoint_dir, "training_curves.png")
    )
 
 
if __name__ == "__main__":
    main()




















    

    










