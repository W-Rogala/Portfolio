"""
Vision Transformer (ViT) model implementation for the profession classifier project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    
    Args:
        img_size (int): Image size. Default: 224
        patch_size (int): Patch token size. Default: 16
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 768
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # B, C, H, W -> B, D, H//P, W//P -> B, D, N -> B, N, D
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """
    Attention mechanism with projection.
    
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: False
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks.
    
    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        out_features (int): Number of output features.
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer Block with self-attention and MLP.
    
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: False
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer model for image classification.
    
    Args:
        img_size (int): Image size. Default: 224
        patch_size (int): Patch size. Default: 16
        in_chans (int): Number of input channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 10
        embed_dim (int): Patch embedding dimension. Default: 768
        depth (int): Depth of transformer. Default: 12
        num_heads (int): Number of attention heads. Default: 12
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=10,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim  # num_features for consistency with other models
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Append cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x[:, 0]  # Return only the cls token features

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_tiny(num_classes=10):
    """
    Create a tiny Vision Transformer model.
    Smaller than standard ViT for efficient training on profession classification.
    """
    return VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_ratio=4,
        num_classes=num_classes
    )


def vit_small(num_classes=10):
    """
    Create a small Vision Transformer model.
    Similar to ViT-S/16 but adapted for profession classification.
    """
    return VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        num_classes=num_classes
    )


def vit_base(num_classes=10):
    """
    Create a base Vision Transformer model.
    Similar to ViT-B/16 but adapted for profession classification.
    """
    return VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=num_classes
    )


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test Vision Transformer models
    models = {
        "ViT-Tiny": vit_tiny(),
        "ViT-Small": vit_small(),
        "ViT-Base": vit_base()
    }
    
    # Print model architectures and parameter counts
    for name, model in models.items():
        X = torch.randn(size=(1, 3, 224, 224))
        output = model(X)
        
        print(f"\n{name} Architecture:")
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Parameter count: {count_parameters(model):,}")
