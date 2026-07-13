import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Sequence
import logging

logger = logging.getLogger(__name__)

class ChestClassifier(nn.Module):
    """Binary chest X-ray classifier using RAD-DINO-MAIRA-2 encoder."""
    
    def __init__(self,
                 model_name: str = "microsoft/rad-dino-maira-2",
                 num_classes: int = 2,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.1,
                 freeze_encoder: bool = False,
                 num_subtypes: int = 0):
        """
        Args:
            model_name: HuggingFace model name for the encoder
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for classifier head
            dropout_rate: Dropout rate
            freeze_encoder: Whether to freeze the encoder weights
            num_subtypes: If > 0, add an auxiliary multi-label head predicting this
                many pathology subtypes, trained jointly with the binary head.
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.num_subtypes = num_subtypes
        self.freeze_encoder = freeze_encoder
        
        # Load the RAD-DINO-MAIRA-2 model
        try:
            self.encoder = AutoModel.from_pretrained(model_name)
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
        
        # Get encoder output dimension
        config = AutoConfig.from_pretrained(model_name)
        self.encoder_dim = config.hidden_size
        
        # Freeze encoder if specified
        if freeze_encoder:
            self.freeze_encoder_weights()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Auxiliary multi-label subtype head (optional). Shares the encoder
        # features; forces the backbone to learn features that separate the
        # targeted pathology subtypes (fractures, OA, dislocation, ...).
        if num_subtypes and num_subtypes > 0:
            self.subtype_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.encoder_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, num_subtypes),
            )
        else:
            self.subtype_head = None

        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier (and subtype) head weights using Xavier init."""
        heads = [self.classifier]
        if self.subtype_head is not None:
            heads.append(self.subtype_head)
        for head in heads:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def freeze_encoder_weights(self):
        """Freeze encoder weights for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True
        logger.info("Encoder weights frozen")
    
    def unfreeze_encoder_weights(self):
        """Unfreeze encoder weights for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
        logger.info("Encoder weights unfrozen")

    def _dinov2_encoder_blocks(self) -> Sequence[nn.Module]:
        """Return the transformer block ModuleList for Dinov2Model (RAD-DINO-MAIRA-2)."""
        enc = self.encoder
        inner = getattr(enc, "encoder", None)
        if inner is not None and hasattr(inner, "layer"):
            return inner.layer
        if hasattr(enc, "layer"):
            return enc.layer
        raise RuntimeError(
            "Could not find encoder blocks (expected Dinov2Model.encoder.layer). "
            f"Encoder type: {type(enc).__name__}"
        )

    def unfreeze_last_n_encoder_blocks(self, n: int = 4) -> None:
        """
        Freeze the entire backbone, then train only the last n transformer blocks.

        Embeddings, final LayerNorm (if outside those blocks), and earlier blocks stay frozen.
        """
        blocks = self._dinov2_encoder_blocks()
        depth = len(blocks)
        if n <= 0:
            self.freeze_encoder_weights()
            return
        if n > depth:
            logger.warning(
                "Requested last %d encoder blocks but only %d exist; using %d.",
                n,
                depth,
                depth,
            )
            n = depth
        self.freeze_encoder_weights()
        start = depth - n
        for block in blocks[start:]:
            for p in block.parameters():
                p.requires_grad = True
        self.freeze_encoder = False
        logger.info(
            "Unfroze encoder blocks %d–%d (last %d of %d)",
            start,
            depth - 1,
            n,
            depth,
        )

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pixel_values: Input images [batch_size, channels, height, width]
            
        Returns:
            Dict containing logits and features
        """
        # Get encoder outputs. interpolate_pos_encoding=True lets the DINOv2 backbone
        # accept inputs larger/smaller than its native 518px (no-op at native size) so
        # high-resolution training works. Fall back gracefully if unsupported.
        try:
            encoder_outputs = self.encoder(pixel_values, interpolate_pos_encoding=True)
        except TypeError:
            encoder_outputs = self.encoder(pixel_values)
        
        # Use CLS token (first token) as feature representation
        # For DINO models, we typically use the pooled output
        if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
            features = encoder_outputs.pooler_output
        else:
            # Fallback to mean pooling of last hidden state
            features = encoder_outputs.last_hidden_state.mean(dim=1)
        
        # Classification
        logits = self.classifier(features)

        out = {
            'logits': logits,
            'features': features
        }
        if self.subtype_head is not None:
            out['subtype_logits'] = self.subtype_head(features)
        return out
    
    def get_encoder_parameters(self):
        """Get encoder parameters for separate optimization."""
        return self.encoder.parameters()
    
    def get_classifier_parameters(self):
        """Get classifier (and subtype head) parameters for separate optimization."""
        import itertools
        if self.subtype_head is not None:
            return itertools.chain(
                self.classifier.parameters(), self.subtype_head.parameters()
            )
        return self.classifier.parameters()
    
    def get_trainable_parameters(self):
        """Get all trainable parameters."""
        return filter(lambda p: p.requires_grad, self.parameters())

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BinaryFocalLossWithLogits(nn.Module):
    """Focal loss for a single-logit binary head (sigmoid).

    Down-weights easy examples so training focuses on the hard, subtle cases
    (subtle OA / AC-arthritis / hairline fractures the model currently misses).
    ``pos_weight`` is applied to positives exactly as in BCEWithLogitsLoss.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('pos_weight', pos_weight if pos_weight is not None else None)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs, targets: (N, 1) float. BCE per-element, then focal modulation.
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        p = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, p, 1 - p)
        loss = (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class AsymmetricLossBinary(nn.Module):
    """Asymmetric loss (ASL) for a single-logit binary head.

    Decouples positive/negative focusing (gamma_pos < gamma_neg) and applies a
    probability shift (clip) to hard negatives, so abundant easy negatives are
    down-weighted harder than the rarer positives — well suited to imbalanced,
    long-tailed medical findings.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0,
                 clip: float = 0.05, reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs, targets: (N, 1) float.
        x_sig = torch.sigmoid(inputs)
        xs_pos = x_sig
        xs_neg = 1.0 - x_sig
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        pt0 = xs_pos * targets
        pt1 = xs_neg * (1 - targets)
        pt = pt0 + pt1
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss = loss * ((1 - pt) ** gamma)
        loss = -loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss."""
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()
        
        # Apply label smoothing
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / inputs.size(-1)
        
        loss = -torch.sum(targets_smooth * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def create_model(config) -> ChestClassifier:
    """Create and return the chest classifier model."""
    model = ChestClassifier(
        model_name=config.model_name,
        num_classes=config.num_classes,
        hidden_dim=config.classifier_hidden_dim,
        dropout_rate=config.dropout_rate,
        freeze_encoder=True,  # Start with frozen encoder
        num_subtypes=int(getattr(config, "num_subtypes", 0) or 0),
    )
    n_unfreeze = int(getattr(config, "encoder_unfreeze_last_n", 0) or 0)
    if n_unfreeze > 0:
        model.unfreeze_last_n_encoder_blocks(n_unfreeze)
    return model


def init_from_chest_checkpoint(model: ChestClassifier, ckpt_path: str,
                               device: str = "cpu") -> Dict[str, int]:
    """Initialize a shoulder model from a chest MAIRA checkpoint.

    Loads the encoder and binary classifier head from the chest model. The
    auxiliary subtype head (if any) and any shape-mismatched tensors are left
    at their fresh initialization. Returns a small summary dict.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model_sd = model.state_dict()
    to_load, skipped_shape = {}, []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            to_load[k] = v
        elif k in model_sd:
            skipped_shape.append(k)
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    # `missing` here = keys in the model not provided by the chest ckpt
    # (e.g. the fresh subtype_head) — expected, not an error.
    summary = {
        "loaded": len(to_load),
        "skipped_shape_mismatch": len(skipped_shape),
        "fresh_in_model": len([m for m in missing if m not in to_load]),
        "unused_in_ckpt": len(unexpected),
    }
    logger.info(
        "init_from_chest: loaded=%d skipped(shape)=%d fresh=%d unused=%d",
        summary["loaded"], summary["skipped_shape_mismatch"],
        summary["fresh_in_model"], summary["unused_in_ckpt"],
    )
    if skipped_shape:
        logger.info("  shape-mismatched (kept fresh): %s", skipped_shape)
    return summary


def get_loss_function(config, class_weights: Optional[torch.Tensor] = None,
                      pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """Binary single-output (num_classes=1) uses weighted BCE, or focal BCE if enabled."""
    if getattr(config, 'num_classes', 2) == 1:
        if getattr(config, 'use_asl', False):
            # Asymmetric loss: stronger down-weighting of easy negatives.
            return AsymmetricLossBinary(
                gamma_neg=float(getattr(config, 'asl_gamma_neg', 4.0)),
                gamma_pos=float(getattr(config, 'asl_gamma_pos', 1.0)),
                clip=float(getattr(config, 'asl_clip', 0.05)),
            )
        if getattr(config, 'use_focal', False):
            # Focal BCE: focus learning on hard/subtle examples.
            return BinaryFocalLossWithLogits(
                gamma=float(getattr(config, 'focal_gamma', 2.0)),
                pos_weight=pos_weight,
            )
        # Binary: single logit, sigmoid > 0.5 = abnormal. Weighted BCE for imbalanced data.
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if config.label_smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    return nn.CrossEntropyLoss(weight=class_weights)