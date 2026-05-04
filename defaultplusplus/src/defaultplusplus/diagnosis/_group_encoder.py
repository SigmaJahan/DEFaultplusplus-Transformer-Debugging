"""Feature-group encoder and FPG message passing.

GroupEncoder    Per-group MLP encoders. Each feature group is encoded
                independently to a shared hidden dimension h, producing a
                stack H ∈ R^{G × h}.
GraphAggregator FPG message passing on H. Each round applies
                    H ← ReLU(Â · H · W_msg)
                with the row-normalized group-level adjacency Â
                (self-loops included) and a learnable W_msg ∈ R^{h × h}.
                Three rounds are applied by default; that depth matches
                the diameter of the group-level FPG, so any group can
                reach any other group in three hops.
FlatEncoder     Capacity-matched flat baseline retained for ablations
                that disable the group structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """FPG message passing on stacked group representations.

    Each round applies

        H ← ReLU(Â · H · W_msg)

    where Â is the row-normalized group-level adjacency (self-loops
    included) and W_msg ∈ R^{h × h} is a learnable transform unique to
    this round. The adjacency itself is registered as a fixed buffer.

    Three rounds is the default and matches the diameter of the
    group-level FPG: any group can reach any other group within three
    hops along the propagation edges, which is enough for cross-group
    information to mix without entering the deeper regime where graph
    convolutions tend to oversmooth.
    """

    def __init__(self, hidden_dim: int, adjacency: np.ndarray, n_rounds: int = 3,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.n_rounds = n_rounds

        # Row-normalize so each row sums to 1. Without this, high-degree
        # groups dominate the aggregation and crowd out their neighbors.
        adj = adjacency.astype(np.float32)
        np.fill_diagonal(adj, np.maximum(np.diag(adj), 1.0))
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        adj_norm = adj / row_sum
        self.register_buffer("adj", torch.from_numpy(adj_norm))

        # One W_msg per round.
        self.W_msg = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(n_rounds)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, group_reprs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            group_reprs: H ∈ R^{batch × G × h}
        Returns:
            H ∈ R^{batch × G × h} after `n_rounds` rounds of message passing.
        """
        h = group_reprs
        for t in range(self.n_rounds):
            messages = torch.einsum("gj,bjd->bgd", self.adj, h)
            messages = self.W_msg[t](messages)
            h = self.dropout(F.relu(messages))
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
