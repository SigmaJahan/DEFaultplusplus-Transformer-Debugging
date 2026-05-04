"""Hierarchical fault-diagnosis model with FPG-based explanation.

Three-level architecture:

  Level 1  Fault detection:      binary clean vs faulty.
  Level 2  Fault categorization: predict the fault category.
  Level 3  Root-cause diagnosis: prototype matcher in the group-embedding
                                 space, with a built-in explanation that
                                 decomposes the prediction across feature
                                 groups.

Levels 1 and 2 read the projected embedding z. Level 3 reads the full
group-embedding stack H so that distances to per-class prototypes
decompose additively across feature groups, which is the basis for the
explanation produced by ``explain_diagnosis``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._group_encoder import GroupEncoder, GraphAggregator


class FlatEncoder(nn.Module):
    """Capacity-matched flat MLP encoder.

    In flat mode, we still produce group-shaped output by splitting the
    hidden representation into n_groups chunks, so Stage 3 prototype
    matching works identically in both modes.
    """

    def __init__(self, input_dim: int, n_groups: int, hidden_dim_per_group: int,
                 embedding_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_groups = n_groups
        self.hidden_dim_per_group = hidden_dim_per_group
        total_hidden = n_groups * hidden_dim_per_group

        self.net = nn.Sequential(
            nn.Linear(input_dim, total_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_hidden, total_hidden),
        )
        self.projection = nn.Linear(total_hidden, embedding_dim)

    def forward(self, x):
        """Returns (z_projected, h_groups).

        z_projected: (batch, embedding_dim) -- for Stage 1, 2
        h_groups:    (batch, n_groups, hidden_dim) -- for Stage 3
        """
        h = self.net(x)  # (batch, n_groups * hidden_dim)
        z = self.projection(h)
        h_groups = h.reshape(x.shape[0], self.n_groups, self.hidden_dim_per_group)
        return z, h_groups


class HierarchicalDiagnosisModel(nn.Module):
    """Full hierarchical model with FPG-based explainability for fault diagnosis.

    encode() returns both:
      - z: (batch, embedding_dim) projected embedding for Stage 1, 2
      - h_groups: (batch, n_groups, hidden_dim) group-level for Stage 3

    Stage 3 uses prototypical classification in the group-structured space.
    Prototypes are computed as mean group embeddings per root-cause class.
    Distance to prototype decomposes by group -> FPG-based explainability.
    """

    def __init__(self, input_dim=None, group_dims=None, adjacency=None,
                 hidden_dim=32, embedding_dim=64, n_message_passing=1,
                 dropout=0.1, mode="flat", n_categories=0,
                 category_sizes=None, group_names=None):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_categories = n_categories
        self.category_sizes = category_sizes or {}

        # Store group names for explanation
        if group_names:
            self._group_names = sorted(group_names)
        elif group_dims:
            self._group_names = sorted(group_dims.keys())
        else:
            self._group_names = [f"group_{i}" for i in range(8)]
        self.n_groups = len(self._group_names)

        # -- Shared backbone -------------------------------------------------
        if mode == "flat":
            assert input_dim is not None
            self.encoder = FlatEncoder(
                input_dim, n_groups=self.n_groups,
                hidden_dim_per_group=hidden_dim,
                embedding_dim=embedding_dim, dropout=dropout)
        elif mode == "graph_conditioned":
            assert group_dims is not None
            self.group_encoder = GroupEncoder(group_dims, hidden_dim=hidden_dim,
                                              dropout=dropout)
            self.projection = nn.Linear(hidden_dim * self.n_groups, embedding_dim)
            if adjacency is not None:
                self.aggregator = GraphAggregator(
                    hidden_dim, adjacency,
                    n_rounds=n_message_passing, dropout=dropout)
            else:
                self.aggregator = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # -- Stage 1: Detection head -----------------------------------------
        self.detection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 2),
        )

        # -- Stage 2: Category head ------------------------------------------
        self.category_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, n_categories),
        )

        # -- Stage 3: Per-category root-cause heads --------------------------
        # These produce logits for CE training. During inference we ALSO use
        # prototype matching in group space for explainability.
        self.rootcause_heads = nn.ModuleDict()
        for cat_name, n_rc in self.category_sizes.items():
            if n_rc >= 2:
                self.rootcause_heads[cat_name] = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(embedding_dim // 2, n_rc),
                )

        # Prototype storage (set after training via compute_prototypes)
        # Shape per category: (n_rootcauses, n_groups, hidden_dim)
        self._prototypes = {}

    @property
    def group_names(self):
        return self._group_names

    def encode(self, x, group_indices=None):
        """Shared backbone -> (z, h_groups).

        Returns:
            z:        (batch, embedding_dim) for Stage 1, 2
            h_groups: (batch, n_groups, hidden_dim) for Stage 3 + explainability
        """
        if self.mode == "flat":
            return self.encoder(x)

        encoded = self.group_encoder.forward_flat(x, group_indices)
        gnames = self.group_encoder.group_names
        h_list = []
        for name in gnames:
            if name in encoded:
                h_list.append(encoded[name])
            else:
                h_list.append(torch.zeros(x.shape[0], self.hidden_dim,
                                          device=x.device))
        h_stack = torch.stack(h_list, dim=1)  # (batch, n_groups, hidden_dim)

        if self.aggregator is not None:
            h_stack = self.aggregator(h_stack)

        h_flat = h_stack.reshape(x.shape[0], -1)
        z = self.projection(h_flat)
        return z, h_stack

    def detect(self, z):
        """Stage 1: (batch, embedding_dim) -> (batch, 2) logits."""
        return self.detection_head(z)

    def categorize(self, z):
        """Stage 2: (batch, embedding_dim) -> (batch, n_categories) logits."""
        return self.category_head(z)

    def diagnose(self, z, category_name):
        """Stage 3 (CE head): (batch, embedding_dim) -> (batch, n_rc) logits."""
        if category_name in self.rootcause_heads:
            return self.rootcause_heads[category_name](z)
        return None

    # -- Prototype-based Stage 3 (for explainability) -------------------------

    def compute_prototypes(self, h_groups, y_local, category_name):
        """Compute and store mean group embeddings per root-cause class.

        Args:
            h_groups: (N, n_groups, hidden_dim) group embeddings for one category
            y_local:  (N,) local root-cause labels (0..n_rc-1)
            category_name: which category these belong to
        """
        n_rc = self.category_sizes.get(category_name, 0)
        if n_rc < 2:
            return
        protos = torch.zeros(n_rc, self.n_groups, self.hidden_dim,
                             device=h_groups.device)
        for c in range(n_rc):
            mask = y_local == c
            if mask.sum() > 0:
                protos[c] = h_groups[mask].mean(dim=0)
        self._prototypes[category_name] = protos.detach()

    def diagnose_proto(self, h_groups, category_name):
        """Prototype-based diagnosis in group-structured space.

        Args:
            h_groups: (batch, n_groups, hidden_dim)
            category_name: which category

        Returns:
            preds:       (batch,) predicted root-cause index
            distances:   (batch, n_rc) total distance to each prototype
            group_dists: (batch, n_rc, n_groups) per-group distance contributions
        """
        if category_name not in self._prototypes:
            return None, None, None

        protos = self._prototypes[category_name]  # (n_rc, n_groups, hidden_dim)
        n_rc = protos.shape[0]
        batch = h_groups.shape[0]

        # Per-group squared euclidean distance: (batch, n_rc, n_groups)
        # h_groups: (batch, 1, n_groups, hidden_dim)
        # protos:   (1, n_rc, n_groups, hidden_dim)
        diff = h_groups.unsqueeze(1) - protos.unsqueeze(0)
        group_dists = (diff ** 2).sum(dim=-1)  # (batch, n_rc, n_groups)

        # Total distance per prototype: (batch, n_rc)
        distances = group_dists.sum(dim=-1)

        # Predict nearest prototype
        preds = distances.argmin(dim=-1)  # (batch,)

        return preds, distances, group_dists

    def explain_diagnosis(self, h_groups, category_name, pred_rc_idx=None):
        """Explain a root-cause prediction by feature-group importance.

        For each sample we compare the predicted prototype to the nearest
        alternative prototype within the same fault category. The squared
        distance to each prototype decomposes additively across groups, so

            delta_g = d_g(nearest_alt) - d_g(predicted)

        is the per-group margin. Positive delta_g means group g pulls the
        sample toward the predicted prototype rather than the alternative
        and therefore supports the prediction. Negative deltas are
        clamped to zero before normalization, so importance scores are
        non-negative and sum to one.

        Args:
            h_groups:    (batch, G, h) group embeddings.
            category_name: fault category whose prototypes to use.
            pred_rc_idx: optional (batch,) predicted root-cause indices;
                         if omitted, predictions are taken from the
                         prototype matcher.

        Returns:
            List of length ``batch``. Each element is a dict mapping
            group name to a non-negative importance score; per sample the
            scores sum to one. Returns ``None`` if no prototypes are
            stored for ``category_name``.
        """
        preds, distances, group_dists = self.diagnose_proto(h_groups, category_name)
        if preds is None:
            return None

        if pred_rc_idx is not None:
            preds = pred_rc_idx

        batch = h_groups.shape[0]
        n_rc = group_dists.shape[1]
        explanations = []

        for i in range(batch):
            rc_pred = int(preds[i].item())
            d_per_proto = distances[i]                  # (n_rc,)

            # Nearest alternative prototype (smallest distance among rivals).
            if n_rc < 2:
                expl = {name: 1.0 / self.n_groups for name in self._group_names}
                explanations.append(expl)
                continue

            mask = torch.ones(n_rc, dtype=torch.bool, device=d_per_proto.device)
            mask[rc_pred] = False
            rivals = d_per_proto.masked_fill(~mask, float("inf"))
            rc_alt = int(rivals.argmin().item())

            # Per-group margin between alternative and predicted.
            d_g_pred = group_dists[i, rc_pred]          # (G,)
            d_g_alt = group_dists[i, rc_alt]            # (G,)
            delta_g = d_g_alt - d_g_pred                # (G,)

            pos = torch.clamp(delta_g, min=0.0)
            denom = pos.sum().item()
            if denom < 1e-10:
                # Degenerate case: no group strictly favors the prediction.
                expl = {name: 1.0 / self.n_groups for name in self._group_names}
            else:
                expl = {name: float(pos[j].item() / denom)
                        for j, name in enumerate(self._group_names)}
            explanations.append(expl)

        return explanations
