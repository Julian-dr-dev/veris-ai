

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from typing import Dict, Tuple, Union
import os
 
from models.ijepa import build_ijepa_small, build_ijepa_base, compute_anomaly_score



def build_transform(img_size: int = 224) -> transforms.Compose: 

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.299, 0.224, 0.225],
        ),
    ])



#heatmap for MRIs
def patch_scores_to_heatmap(
    patch_scores: torch.Tensor,
    patch_grid: int,
    img_size: int = 224,
) -> np.ndarray:
    
    scores = patch_scores.cpu().numpy()

    #reshapes the flat list of scores into a 2d patch grid
    grid = scores.reshape(patch_grid, patch_grid)

    min_val = grid.min()
    max_val = grid.max()
    if max_val > min_val:
        grid = (grid - min_val) / (max_val - min_val)
    else: 
        grid = np.zeros_like(grid)

    

    heatmap_gray = cv2.resize(
        grid.astype(np.float32),
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_uint8 = (heatmap_gray * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    
    return heatmap_rgb


