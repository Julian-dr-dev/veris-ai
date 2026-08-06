

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


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    

    #resize original image to match the heatmap
    img_size = heatmap.shape[0]
    original_resized = original_image.resize(
        (img_size, img_size),
        Image.BILINEAR,
    ).convert("RGB")


    original_np = np.array(original_resized).astype(np.float32)
    heatmap_np = heatmap.astype(np.float32)

    blended = (1 - alpha) * original_np + alpha * heatmap_np
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)





#actual detector class: 
class VerisDetector:

    def __init__(
        self,
        checkpoint_path: str,
        variant:         str   = "small",
        img_size:        int   = 224,
        threshold:       float = None,
        n_masks:         int   = 10,
        device:          str   = None,
    ):
        
        self.img_size = img_size
        self.n_masks = n_masks


        if device is None: 
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else: 
            self.device = device

        print(f"[detector] Loading model from {checkpoint_path}")
        print(f"[detector] Device: {self.device}")


        if variant == "small":
            self.model = build_ijepa_small(img_size=img_size)
        else:
            self.model = build_ijepa_base(img_size=img_size)

        

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Run train.py first to generate a checkpoint."
            )
        

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model= self.model.to(self.device)
        self.model.eval()


        trained_epoch = ckpt.get("epoch", "unknown")
        trained_loss  = ckpt.get("loss",  "unknown")
        print(f"[detector] Loaded checkpoint — epoch {trained_epoch}, loss {trained_loss:.4f}")


        self.transform = build_transform(img_size)
        self.threshold = threshold if threshold is not None else 0.002

        print(f"[detector] Ready — threshold={self.threshold:.4f}")
    


    def preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:

        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Expected str or PIL Image, got {type(image)}")
        

        self._last_original = img

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor
    

    def predict(
        self, 
        image: Union[str, Image.Image], 
        return_heatmap: bool = True,
    
    ) -> Dict:
        
        tensor = self.preprocess(image)

        anomaly_score, patch_scores = compute_anomaly_score(
            self.model,
            tensor,
            n_masks=self.n_masks,
        )

        is_anomalous = anomaly_score > self.threshold


        result = {
            "anomaly_score": anomaly_score,
            "patch_scores":  patch_scores,
            "is_anomalous":  is_anomalous,
            "threshold":     self.threshold,
        }

        if return_heatmap:
            heatmap_np = patch_scores_to_heatmap(
                patch_scores,
                patch_grid=self.model.patch_grid,
                img_size=self.img_size,
            )
            heatmap_pil = Image.fromarray(heatmap_np)
            overlay_pil = overlay_heatmap(self._last_original, heatmap_np)
            original_resized = self._last_original.resize(
                (self.img_size, self.img_size),
                Image.BILINEAR,
            )
 
            result["heatmap"]  = heatmap_pil
            result["overlay"]  = overlay_pil
            result["original"] = original_resized
 
        return result
    


    def calibrate_threshold(
        self, 
        normal_image_dir: str,
        percentile: float = 95.0,      
    ) -> float:
        
        print(f"[detector] Calibrating threshold on {normal_image_dir}...")

        scores = []
        extensions = {".png", ".jpg", ".jpeg", ".bmp"}


        image_files = [
            os.path.join(normal_image_dir, f)
            for f in sorted(os.listdir(normal_image_dir))
            if os.path.splitext(f)[1].lower() in extensions
        ]

        if not image_files: 
            raise FileNotFoundError(f"No images found in {normal_image_dir}")
        
        for path in image_files:
            result = self.predict(path, return_heatmap=False)
            scores.append(result["anamoly_score"])
        

        scores_np = np.array(scores)
        self.threshold = float(np.percentile(scores_np, percentile))


        print(f"[detector] Calibrated threshold: {self.threshold:.6f}")
        print(f"[detector] Score range: {scores_np.min():.6f} — {scores_np.max():.6f}")
        print(f"[detector] Score mean:  {scores_np.mean():.6f}")
 
        return self.threshold
        


















#one test: 
if __name__ == "__main__":
    import sys
 
    checkpoint = "data/checkpoints/ijepa_best.pth"
 
    # Load detector
    detector = VerisDetector(checkpoint, variant="small")
 
    # Calibrate threshold on normal training images
    detector.calibrate_threshold("data/mvtec/bottle/train/good")
 
    print(f"\nCalibrated threshold: {detector.threshold:.6f}")
 
    # Test on a normal image
    print("\n--- Testing on normal image ---")
    normal_img = "data/mvtec/bottle/test/good/000.png"
    if os.path.exists(normal_img):
        result = detector.predict(normal_img)
        print(f"Score:        {result['anomaly_score']:.6f}")
        print(f"Is anomalous: {result['is_anomalous']}")
        result["overlay"].save("data/checkpoints/normal_overlay.png")
        print(f"Overlay saved → data/checkpoints/normal_overlay.png")
 
    # Test on a defective image
    print("\n--- Testing on defective image ---")
    defect_img = "data/mvtec/bottle/test/broken_large/000.png"
    if os.path.exists(defect_img):
        result = detector.predict(defect_img)
        print(f"Score:        {result['anomaly_score']:.6f}")
        print(f"Is anomalous: {result['is_anomalous']}")
        result["overlay"].save("data/checkpoints/defect_overlay.png")
        print(f"Overlay saved → data/checkpoints/defect_overlay.png")



