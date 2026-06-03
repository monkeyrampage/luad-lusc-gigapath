"""
model.py
--------
Models for LUAD/LUSC classification from GigaPath embeddings.

  ABMIL       : Attention-Based MIL (Ilse et al. 2018) — standard
  GatedABMIL  : Gated attention variant
  MeanPoolMLP : Mean-pool baseline + MLP (no attention)

All models accept a variable-length bag (N_tiles, 1536) and output
a (2,) logit vector for binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """
    Standard Attention-Based MIL.
    Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.

    Architecture:
      tile embedding (1536) → projection (L) → attention score → weighted sum → classifier
    """

    def __init__(self, input_dim=1536, hidden_dim=512, n_classes=2, dropout=0.25):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, bag):
        """
        Args:
            bag: (N_tiles, input_dim) tensor
        Returns:
            logits:   (n_classes,) tensor
            attn_w:   (N_tiles,)   attention weights (for visualization)
        """
        H = self.projection(bag)           # (N, hidden_dim)
        A = self.attention(H)              # (N, 1)
        A = F.softmax(A, dim=0)            # (N, 1) — sum to 1 over tiles
        z = (A * H).sum(dim=0)            # (hidden_dim,) weighted sum
        logits = self.classifier(z)        # (n_classes,)
        return logits, A.squeeze(-1)       # attn_w: (N,)


class GatedABMIL(nn.Module):
    """
    Gated Attention-Based MIL.
    Introduces a sigmoid gate for improved gradient flow.
    tanh(V·h) ⊙ σ(U·h) replaces tanh(V·h) in the attention branch.
    """

    def __init__(self, input_dim=1536, hidden_dim=512, n_classes=2, dropout=0.25):
        super().__init__()

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        # Gated attention — two parallel branches
        self.attn_V = nn.Linear(hidden_dim, hidden_dim // 2)   # tanh branch
        self.attn_U = nn.Linear(hidden_dim, hidden_dim // 2)   # sigmoid gate
        self.attn_w = nn.Linear(hidden_dim // 2, 1)            # scalar score

        # Classifier
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, bag):
        """
        Args:
            bag: (N_tiles, input_dim) tensor
        Returns:
            logits:  (n_classes,)
            attn_w:  (N_tiles,) attention weights
        """
        H = self.projection(bag)                                    # (N, hidden_dim)
        gate = torch.tanh(self.attn_V(H)) * torch.sigmoid(self.attn_U(H))  # (N, hidden_dim//2)
        A = self.attn_w(gate)                                       # (N, 1)
        A = F.softmax(A, dim=0)                                     # (N, 1)
        z = (A * H).sum(dim=0)                                      # (hidden_dim,)
        logits = self.classifier(z)                                  # (n_classes,)
        return logits, A.squeeze(-1)


class MeanPoolMLP(nn.Module):
    """
    Baseline: mean-pool GigaPath embeddings → MLP classifier.
    No attention — tests whether attention adds value over simple pooling.
    """

    def __init__(self, input_dim=1536, hidden_dim=256, n_classes=2, dropout=0.25):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, bag):
        """
        Args:
            bag: (N_tiles, input_dim) tensor
        Returns:
            logits:  (n_classes,)
            attn_w:  (N_tiles,) uniform weights (for API compatibility)
        """
        z      = bag.mean(dim=0)           # (input_dim,) simple mean pool
        logits = self.mlp(z)               # (n_classes,)
        attn_w = torch.ones(len(bag), device=bag.device) / len(bag)
        return logits, attn_w


def get_model(name, **kwargs):
    """
    Factory function.
    name: 'abmil' | 'gated_abmil' | 'meanpool_mlp'
    """
    models = {
        "abmil":        ABMIL,
        "gated_abmil":  GatedABMIL,
        "meanpool_mlp": MeanPoolMLP,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(models.keys())}")
    return models[name](**kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bag    = torch.randn(500, 1536).to(device)

    for name in ["abmil", "gated_abmil", "meanpool_mlp"]:
        model  = get_model(name).to(device)
        logits, attn = model(bag)
        print(f"{name:20s} | params={count_parameters(model):>8,} | "
              f"logits={logits.shape} | attn={attn.shape} | "
              f"attn_sum={attn.sum().item():.4f}")
