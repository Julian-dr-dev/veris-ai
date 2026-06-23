

"""
I-JEPA: Image Joint Embedding Predictive Architecture

"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from typing import List, Tuple, Optional




#Patch embedding:
# ─────────────────────────────────────────────────────────────────────────────


class PatchEmbedding(nn.Module):


    def __init__(
            self, img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            embed_dim: int = 768,
        
    ):
        super().__int__()

        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2


        def forward(self, x: torch.Tensor) -> torch.Tensor:

            x = self.projection(x)
            x = x.flatten(2)

            x = x.transpose(1, 2)

            return x



 




class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout:   float = 0.0,
    ):
        super().__init__()
 
        assert embed_dim % num_heads == 0, \
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
 
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5  # scaling factor
 
        # Q, K, V
        self.qkv     = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)





    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, N, C = x.shape
        
        okv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        
        q, k, v = qkv.unbind(0)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)


        x = attn @ v

        x = x.transpose(1, 2).reshape(B, C, C)

        x = self.proj(x)

        return x 
    


 # TRANSFORMER BLOCK

class TransformBlock(nn.Module):



    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.0,
            
    ):
    
        super().__init__()

        mlp_hidden = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))

        x = x + self.mlp(self.norm2(x))

        return x
    


#VISION TRANSFORMER ENCODER

class VisionTransformerEncoder(nn.Module):
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim:   int   = 768,
        depth:       int   = 12,
        num_heads:   int   = 12,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,

    ):
        

        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim

        )

        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
            )
        self.pos_drop = nn.Dropout(dropout)



        self.blocks = nn.ModuleList([
            TransformBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)



    def forward(
        self,
        x: torch.Tensor,
        patch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)


        if patch_indices is not None:
            x = x[:, patch_indices, :]


        for block in self.blocks:
            x = block(x)


        x = self.norm(x)
        return x
    




class Predictor(nn.Module): 

    
     
    def __init__(
        self,
        embed_dim:     int = 768,
        predictor_dim: int = 384,
        num_patches:   int = 196,
        depth:         int = 6,
        num_heads:     int = 12,
    ):
        super().__init__()


        self.input_proj = nn.Linear(embed_dim, predictor_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_dim)
        )



        self.blocks = nn.ModuleList([
            TransformBlock(predictor_dim, num_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_dim)


        self.output_proj = nn.Linear(predictor_dim, embed_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
            self,
            context_embeddings: torch.Tensor,
            context_indices: torch.Tensor,
            target_indices: torch.Tensor,

    ) -> torch.Tensor:
        
        
        B = context_embeddings.shape[0]

        x_context = self.input_proj(context_embeddings)


        x_context = x_context + self.pos_embed[:, context_indices, :]

        n_target = len(target_indices)
        mask_tokens = self.mask_token.expand(B, n_target, -1)

        mask_tokens = mask_tokens + self.pos_embed[:, target_indices, :]


        x = torch.cat([x_context, mask_tokens], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)


        #grabbing the last target tokens from the sequence:
        x = x[:, -n_target:, :]

        x = self.output_proj(x)


        return x 






#masking:

def create_masks(
        num_patches: int,
        num_target_blocks: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.2),
        target_ratio:      Tuple[float, float] = (0.75, 1.5),
        context_scale:     float = 0.85,
        patch_grid:        int = 14,

        
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
    all_patches = set(range(num_patches))
    target_patches = set()
    target_blocks = []
 
    for _ in range(num_target_blocks):
        # Sample block scale and aspect ratio
        scale = np.random.uniform(*target_scale)
        ratio = np.random.uniform(*target_ratio)
 
        # Compute block dimensions in patch units
        area   = scale * num_patches
        height = int(round(np.sqrt(area / ratio)))
        width  = int(round(np.sqrt(area * ratio)))
 
        # Clamp to grid size
        height = max(1, min(height, patch_grid))
        width  = max(1, min(width,  patch_grid))
 
        # Sample random top-left corner
        top  = np.random.randint(0, patch_grid - height + 1)
        left = np.random.randint(0, patch_grid - width  + 1)
 
        # Collect patch indices in this block
        block_indices = []
        for r in range(top, top + height):
            for c in range(left, left + width):
                idx = r * patch_grid + c
                block_indices.append(idx)
                target_patches.add(idx)
 
        target_blocks.append(torch.tensor(block_indices, dtype=torch.long))
 
    # Context = sample from non-target patches
    non_target = list(all_patches - target_patches)
    n_context  = max(1, int(len(non_target) * context_scale))
    context_indices = torch.tensor(
        np.random.choice(non_target, n_context, replace=False),
        dtype=torch.long,
    )
 
    return context_indices, target_blocks






#full JEPA model:
class IJEPA(nn.Module):




    def __init__(
        self,
        img_size:      int   = 224,
        patch_size:    int   = 16,
        in_channels:   int   = 3,
        embed_dim:     int   = 768,
        encoder_depth: int   = 12,
        encoder_heads: int   = 12,
        pred_dim:      int   = 384,
        pred_depth:    int   = 6,
        pred_heads:    int   = 12,
        ema_decay:     float = 0.996,
):
        

        super().__init__()

        self.ema_decay = ema_decay
        num_patches = (img_size //patch_size) ** 2
        self.patch_grid = img_size // patch_size


        self.context_encoder = VisionTransformerEncoder(
            img_size, patch_size, in_channels,
            embed_dim, encoder_depth, encoder_heads
        )
        self.target_encoder = VisionTransformerEncoder(
            img_size, patch_size, in_channels,
            embed_dim, encoder_depth, encoder_heads
        )


        #duplicator of encoders: 

        self._copy_weights(self.context_encoder, self.target_encoder)
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        
        #predictor:
        self.predictor = Predictor(
            embed_dim, pred_dim, num_patches, pred_depth, pred_heads
        )


        #copy function
    def _copy_weights(self, src: nn.Module, dst: nn.Module):
        for src_param, dst_param in zip(src.parameters(), dst.parameters()):
            dst_param.data.copy_(src_param.data)

        

        #updater:
    def update_target_encoder(self):

        for ctx_param, tgt_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            tgt_param.data = (
                self.ema_decay * tgt_param.data
                + (1.0 - self.ema_decay) * ctx_param.data
            )
    

    def forward(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
        target_indeces: List[torch.Tensor],

        
    ) -> torch.Tensor:
        


        context_embeddings = self.context_encoder(x, context_indices)

        with torch.no_grad():
            
            all_target_embeddings = self.target_encoder(x)
        
        total_loss = 0.0
        n_blocks = len(target_indeces)

        for block_indices in target_indeces:
            block_indices = block_indices.to(x.device)

            target = all_target_embeddings[:, block_indices, :]

            predictions = self.predictor(
                context_embeddings,
                context_indices,
                block_indices,
            )

            predictions = F.normalize(predictions, dim=-1)
            targets = F.normalize(targets, dim=-1)

            loss = F.mse_loss(predictions, targets)
            total_loss += loss

        return total_loss / n_blocks
    




#anamoly scoring: 
@torch.no_grad()

def compute_anamoly_score(
    model: IJEPA, 
    x: torch.Tensor,
    n_masks: int = 10,

) -> Tuple[float, torch.Tensor]:
    


    model.eval()
    device = x.device
    num_patches = model.patch_grid ** 2

    patch_errors = torch.zeros(num_patches, device=device)
    patch_counts = torch.zeros(num_patches, device=device)

    for _ in range(n_masks):
        context_indices, target_blocks, = create_masks(
            num_patches,
            patch_grid=model.patch_grid,
        )

        context_indices = context_indices.to(device)

        context_embs = model.context_encoder(x, context_indices)
        all_target_embs = model.target_encoder(x)


        for block_indices in target_blocks:
            block_indices = block_indices.to(device)

            target = all_target_embs[:, block_indices, :]

            predictions = model.predictor(
                context_embs, context_indices, block_indices
                
            )

            predictions = F.normalize(predictions, dim=-1)
            targets = F.normalize(targets, dim=-1)

            #error-per-patch
            errors = ((predictions - targets) ** 2).mean(dim=-1).squeeze(0)

            patch_errors[block_indices] += errors
            patch_counts[block_indices] += 1



    mask = patch_counts > 0
    patch_scores = torch.zeros(num_patches, device=device)
    patch_scores[mask] = patch_errors[mask] / patch_counts[mask]

    anomaly_score = patch_scores.mean().item()

    return anomaly_score, patch_scores



        
        
























#model factory:

def build_ijepa_small(img_size: int = 224) -> IJEPA:
    """ViT-Small backbone — faster training, good for development."""
    return IJEPA(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        encoder_depth=12,
        encoder_heads=6,
        pred_dim=192,
        pred_depth=6,
        pred_heads=6,
    )
 
 
def build_ijepa_base(img_size: int = 224) -> IJEPA:
    """ViT-Base backbone — standard I-JEPA configuration."""
    return IJEPA(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        pred_dim=384,
        pred_depth=6,
        pred_heads=12,
    )
 
 
def load_ijepa(checkpoint_path: str, variant: str = "small",
               img_size: int = 224, device: str = "cpu") -> IJEPA:
    """Load a trained I-JEPA model from checkpoint."""
    if variant == "small":
        model = build_ijepa_small(img_size)
    else:
        model = build_ijepa_base(img_size)
 
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[ijepa] Loaded checkpoint from {checkpoint_path}")
    return model


        





























# 10. Quick sanity check:
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
 
    # Build small model for quick test
    model = build_ijepa_small(img_size=224).to(device)
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
 
    # Fake batch of 2 images
    x = torch.randn(2, 3, 224, 224).to(device)
 
    # Create masks
    num_patches     = model.patch_grid ** 2
    ctx_idx, tgt_blocks = create_masks(num_patches, patch_grid=model.patch_grid)
    ctx_idx = ctx_idx.to(device)
 
    print(f"Context patches: {len(ctx_idx)}")
    print(f"Target blocks:   {len(tgt_blocks)}")
    print(f"Target block sizes: {[len(b) for b in tgt_blocks]}")
 
    # Forward pass
    loss = model(x, ctx_idx, tgt_blocks)
    print(f"Loss: {loss.item():.4f}")
 
    # EMA update
    model.update_target_encoder()
    print("EMA update: OK")
 
    # Anomaly score
    score, patch_scores = compute_anomaly_score(model, x[:1])
    print(f"Anomaly score: {score:.4f}")
    print(f"Patch scores shape: {patch_scores.shape}")
    print("\nAll checks passed!")