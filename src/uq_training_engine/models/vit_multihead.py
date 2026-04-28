import timm
import torch
import torch.nn as nn


class ViTMultiHead(nn.Module):
    """
    ViT backbone (timm) with a small MLP head predicting six urban-quality scores.

    Concatenates CLS and mean patch embeddings; optional freezing of backbone weights.

    Example::
        In: ViTMultiHead("vit_base_patch14_dinov2.lvd142m", freeze=False)
        Out: module with ``backbone`` and ``head`` submodules
    """

    def __init__(self, model_name: str, freeze: bool = False) -> None:
        """
        Create the timm model and a two-tower MLP mapping ``2 * feat_dim -> 6`` outputs.

        :param model_name: timm model identifier (``num_classes=0``, feature extractor).
        :param freeze: If True, backbone parameters are not trainable.
        :return: None.

        Example::
            In: ViTMultiHead("vit_base_patch14_dinov2.lvd142m", freeze=True)
            Out: ViTMultiHead instance
        """
        super().__init__()
        # num_classes=0: feature extractor; dynamic_img_size allows non-fixed input sizes at inference.
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, dynamic_img_size=True)
        feat_dim = int(self.backbone.num_features)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 6),
        )
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: backbone features, CLS + mean-pool fusion, six-head scores.

        :param x: Batch of images ``(N, C, H, W)``.
        :return: Tensor ``(N, 6)`` — one score per category per image.

        Example::
            In: model(batch_tensor)
            Out: tensor of shape (32, 6) for batch size 32
        """
        features = self.backbone.forward_features(x)
        # Fuse CLS and mean patch embedding (common ViT variant when a single vector is needed).
        cls_token = features[:, 0]
        patch_tokens = features[:, 1:].mean(dim=1)
        combined = torch.cat([cls_token, patch_tokens], dim=1)
        return self.head(combined)
