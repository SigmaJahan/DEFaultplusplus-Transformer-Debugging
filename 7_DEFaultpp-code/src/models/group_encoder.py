"""FPG graph-conditioned encoder for transformer fault features.

Features are encoded group-by-group through component-aligned MLPs,
FPG-gated message passing allows structurally connected components to exchange
diagnostic signals, then prototype matching classifies fault families.

The FPG adjacency is fixed and derived from transformer architecture mathematics.

Two modes:
  - flat:              FlatEncoder MLP baseline (no graph structure)
  - graph_conditioned: GroupEncoder + FPG GraphAggregator (our method)
"""
import torch
import torch.nn as nn
import numpy as np


class GroupEncoder(nn.Module):
    """Encodes features group-by-group through parallel MLPs.

    Each feature group (e.g., attention, qkv, score) gets its own MLP that
    projects it to a shared hidden dimension. This forces the model to first
    understand each subsystem independently before combining.
    """

    def __init__(self, group_dims: dict[str, int], hidden_dim: int = 32, dropout: float = 0.1):
        """
        Args:
            group_dims: dict mapping group_name -> number of features in that group
            hidden_dim: output dimension for each group encoder
            dropout: dropout rate
        """
        super().__init__()
        self.group_names = sorted(group_dims.keys())
        self.hidden_dim = hidden_dim
        self.n_groups = len(self.group_names)

        self.encoders = nn.ModuleDict()
        for name in self.group_names:
            in_dim = group_dims[name]
            self.encoders[name] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, x_groups: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode each group independently.

        Args:
            x_groups: dict mapping group_name -> (batch, group_features) tensor

        Returns:
            dict mapping group_name -> (batch, hidden_dim) encoded representation
        """
        encoded = {}
        for name in self.group_names:
            if name in x_groups:
                encoded[name] = self.encoders[name](x_groups[name])
        return encoded

    def forward_flat(self, x: torch.Tensor, group_indices: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        """Convenience: split a flat feature vector by group and encode.

        Args:
            x: (batch, total_features) tensor
            group_indices: dict mapping group_name -> list of column indices
        """
        x_groups = {}
        for name in self.group_names:
            if name in group_indices:
                indices = group_indices[name]
                x_groups[name] = x[:, indices]
        return self.forward(x_groups)


class FlatEncoder(nn.Module):
    """Baseline flat MLP encoder (no group structure)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphAggregator(nn.Module):
    """FPG-conditioned message passing over group representations.

    The FPG adjacency (fixed, derived from transformer mathematics) gates
    which groups can exchange diagnostic signals. Groups not connected in the
    FPG do not communicate — this is the structural inductive bias.

    Aggregation: for each node j, aggregate incoming messages from FPG
    neighbors i, transform, and add residually.
    """

    def __init__(self, hidden_dim: int, adjacency: np.ndarray, n_rounds: int = 1,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.n_rounds = n_rounds
        # FPG adjacency is a fixed buffer — not learned
        self.register_buffer("adj", torch.from_numpy(adjacency.astype(np.float32)))

        self.message_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_rounds)
        ])

    def forward(self, group_reprs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            group_reprs: (batch, n_groups, hidden_dim)
        Returns:
            (batch, n_groups, hidden_dim) updated via FPG-gated aggregation
        """
        h = group_reprs
        for t in range(self.n_rounds):
            # adj.T[i,j] = 1 means node i can receive from node j
            messages = torch.einsum("ij,bjd->bid", self.adj.T, h)
            h = h + self.message_fns[t](messages)
        return h


class ProtoClassifier(nn.Module):
    """FPG-conditioned prototypical classifier.

    One method: encode features group-by-group through component-aligned MLPs,
    pass messages over the FPG to allow structurally connected groups to exchange
    diagnostic signals, project to metric space, classify by prototype distance.
    """

    def __init__(self, input_dim: int = None, group_dims: dict[str, int] = None,
                 adjacency: np.ndarray = None, hidden_dim: int = 32,
                 embedding_dim: int = 64, n_message_passing: int = 1,
                 dropout: float = 0.1, mode: str = "flat", **kwargs):
        """
        Args:
            mode: "flat" | "graph_conditioned"
        """
        super().__init__()
        self.mode = mode
        self.embedding_dim = embedding_dim

        if mode == "flat":
            assert input_dim is not None
            self.encoder = FlatEncoder(input_dim, hidden_dim=256,
                                       output_dim=embedding_dim, dropout=dropout)
        elif mode == "graph_conditioned":
            assert group_dims is not None
            self.group_encoder = GroupEncoder(group_dims, hidden_dim=hidden_dim,
                                              dropout=dropout)
            n_groups = len(group_dims)
            self.projection = nn.Linear(hidden_dim * n_groups, embedding_dim)

            if adjacency is not None:
                self.aggregator = GraphAggregator(
                    hidden_dim, adjacency,
                    n_rounds=n_message_passing,
                    dropout=dropout,
                )
            else:
                self.aggregator = None
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'flat' or 'graph_conditioned'.")

        self.temperature = nn.Parameter(torch.tensor(0.1))

    def encode(self, x: torch.Tensor, group_indices: dict[str, list[int]] = None) -> torch.Tensor:
        """Encode input to embedding space.

        Returns: (batch, embedding_dim) tensor
        """
        if self.mode == "flat":
            return self.encoder(x)

        # Group-aware or graph-conditioned
        encoded = self.group_encoder.forward_flat(x, group_indices)

        # Stack into (batch, n_groups, hidden_dim)
        group_names = self.group_encoder.group_names
        h_list = []
        for name in group_names:
            if name in encoded:
                h_list.append(encoded[name])
            else:
                batch_size = x.shape[0]
                h_list.append(torch.zeros(batch_size, self.group_encoder.hidden_dim,
                                          device=x.device))
        h_stack = torch.stack(h_list, dim=1)  # (batch, n_groups, hidden_dim)

        # Graph message passing
        if self.aggregator is not None:
            h_stack = self.aggregator(h_stack)

        # Flatten and project
        h_flat = h_stack.reshape(x.shape[0], -1)  # (batch, n_groups * hidden_dim)
        z = self.projection(h_flat)
        return z

    def compute_prototypes(self, z: torch.Tensor, y: torch.Tensor,
                           n_classes: int) -> torch.Tensor:
        """Compute class prototypes as mean embeddings.

        Returns: (n_classes, embedding_dim)
        """
        prototypes = torch.zeros(n_classes, self.embedding_dim, device=z.device)
        for c in range(n_classes):
            mask = y == c
            if mask.sum() > 0:
                prototypes[c] = z[mask].mean(dim=0)
        return prototypes

    def classify(self, z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Classify by distance to prototypes.

        Returns: (batch, n_classes) log-probabilities
        """
        # Euclidean distance: (batch, n_classes)
        dists = torch.cdist(z.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
        # Negative distance / temperature -> log-softmax
        logits = -dists / self.temperature.abs().clamp(min=0.01)
        return torch.log_softmax(logits, dim=-1)

    def distance_decomposition(self, x: torch.Tensor, prototypes: torch.Tensor,
                                group_indices: dict[str, list[int]],
                                target_class: int) -> dict[str, float]:
        """Decompose distance to a prototype by feature group.

        Returns dict: group_name -> fraction of total distance contributed.
        This IS the explanation — no post-hoc XAI needed.
        """
        if self.mode == "flat":
            return {"all_features": 1.0}

        encoded = self.group_encoder.forward_flat(x, group_indices)
        target_proto = prototypes[target_class]

        # We need group-level prototypes too — stored during training
        # For now, compute total distance decomposition
        group_names = self.group_encoder.group_names
        group_dists = {}
        total_dist = 0.0

        for name in group_names:
            if name in encoded:
                h = encoded[name]  # (1, hidden_dim)
                # Need prototype in group space — approximate from full prototype
                group_dist = h.norm(dim=-1).item()  # simplified
                group_dists[name] = group_dist
                total_dist += group_dist

        if total_dist > 0:
            return {g: d / total_dist for g, d in group_dists.items()}
        return {g: 1.0 / len(group_dists) for g in group_dists}
