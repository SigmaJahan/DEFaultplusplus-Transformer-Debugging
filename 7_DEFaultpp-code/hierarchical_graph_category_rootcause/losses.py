"""Loss functions for hierarchical fault diagnosis.

Five losses combined:
  1. Detection loss           -- weighted CE for clean vs faulty (all samples)
  2. Category loss            -- CE for fault family prediction (faulty samples only)
  3. Root-cause CE loss       -- CE for exact root cause within the true category
  4. Intra-family contrastive -- pushes apart similar root causes within same category
  5. Prototype loss           -- NLL from prototype distances in group-structured space,
                                 ensuring h_groups is directly trained for the
                                 representation that generates explanations

Total loss:
  L = L_detect + alpha * L_category + lambda_ * L_rootcause
    + beta * L_contrastive + gamma * L_prototype
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def detection_loss(logits, y_binary, class_weights=None):
    """Stage 1: Weighted cross-entropy for clean (0) vs faulty (1)."""
    return F.cross_entropy(logits, y_binary, weight=class_weights)


def category_loss(logits, y_category):
    """Stage 2: Cross-entropy over fault categories (faulty samples only).

    Uses ignore_index=-1 as safety net: if any clean sample with y_category=-1
    leaks through the caller's filter, it will be ignored rather than crash.
    """
    return F.cross_entropy(logits, y_category, ignore_index=-1)


def rootcause_loss(logits, y_rootcause):
    """Stage 3: Cross-entropy over root causes within one category."""
    return F.cross_entropy(logits, y_rootcause)


def prototype_loss(h_groups, y_rc_local, n_rc_classes, temperature=0.1):
    """Prototype-based NLL loss in group-structured embedding space.

    Computes class prototypes as mean group embeddings, then classifies each
    sample by negative squared distance to prototypes. This directly trains
    h_groups for the same distance decomposition used in explanations.

    Args:
        h_groups: (N, n_groups, hidden_dim) group-level embeddings
        y_rc_local: (N,) local root-cause labels (0..n_rc-1)
        n_rc_classes: number of root-cause classes in this category
        temperature: scaling for distance -> logits conversion
    """
    N = h_groups.shape[0]
    if N < 2 or len(y_rc_local.unique()) < 2:
        return (h_groups * 0).sum()  # zero loss, connected to computation graph

    # Compute prototypes: mean group embedding per root-cause class
    protos = []
    for c in range(n_rc_classes):
        mask = y_rc_local == c
        if mask.sum() > 0:
            protos.append(h_groups[mask].mean(dim=0))
        else:
            protos.append(torch.zeros_like(h_groups[0]))
    protos = torch.stack(protos, dim=0)  # (n_rc, n_groups, hidden_dim)

    # Per-group squared distance: (N, n_rc, n_groups)
    diff = h_groups.unsqueeze(1) - protos.unsqueeze(0)
    group_dists = (diff ** 2).sum(dim=-1)

    # Total distance -> negative logits (closer = higher score)
    distances = group_dists.sum(dim=-1)  # (N, n_rc)
    logits = -distances / max(temperature, 0.01)

    return F.cross_entropy(logits, y_rc_local)


def sibling_separation_loss(z, y_rootcause, temperature=0.1):
    """Intra-family contrastive loss within one fault category.

    Supervised contrastive loss restricted to root causes within the
    same fault family. Operates on embeddings, not logits.

    Args:
        z: (N, D) embeddings of samples from ONE category
        y_rootcause: (N,) root-cause labels (local indices)
        temperature: scaling temperature for similarity
    """
    N = z.shape[0]
    if N < 2:
        return (z * 0).sum()  # zero loss, connected to graph

    unique_labels = y_rootcause.unique()
    if len(unique_labels) < 2:
        return (z * 0).sum()

    z_norm = F.normalize(z, dim=1)
    sim = torch.mm(z_norm, z_norm.t()) / temperature

    labels_eq = y_rootcause.unsqueeze(0) == y_rootcause.unsqueeze(1)
    self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
    pos_mask = labels_eq & ~self_mask

    # Numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim)
    denom = (exp_sim * (~self_mask).float()).sum(dim=1, keepdim=True).clamp(min=1e-8)
    log_prob = sim - torch.log(denom)

    n_pos = pos_mask.float().sum(dim=1)
    has_pos = n_pos > 0
    if has_pos.sum() == 0:
        return (z * 0).sum()

    loss_per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos.clamp(min=1)
    return loss_per_anchor[has_pos].mean()


def compute_detection_weights(y_detect):
    """Compute inverse-frequency class weights for detection loss."""
    n_total = len(y_detect)
    n_clean = (y_detect == 0).sum()
    n_faulty = (y_detect == 1).sum()
    if n_clean == 0 or n_faulty == 0:
        return None
    w_clean = n_total / (2.0 * n_clean)
    w_faulty = n_total / (2.0 * n_faulty)
    return torch.tensor([w_clean, w_faulty], dtype=torch.float32)


def hierarchical_loss(model, z, h_groups, y_detect, y_category, y_rootcause,
                      category_names, rootcause_local_labels,
                      group_indices=None,
                      alpha=1.0, lambda_=1.0, beta=0.5, gamma=0.3,
                      sibling_temperature=0.1,
                      detection_weights=None):
    """Combined hierarchical loss.

    Args:
        model: HierarchicalDiagnosisModel
        z: (N, D) projected embeddings from shared backbone
        h_groups: (N, n_groups, hidden_dim) group-level embeddings
        y_detect: (N,) binary labels (0=clean, 1=faulty)
        y_category: (N,) category indices (-1 for clean samples)
        y_rootcause: (N,) global root-cause indices (-1 for clean/invalid)
        category_names: list of category name strings
        rootcause_local_labels: dict category_name -> {global_rc_idx: local_idx}
        alpha: weight for category loss
        lambda_: weight for root-cause CE loss
        beta: weight for intra-family contrastive loss
        gamma: weight for prototype loss (trains h_groups for explanation fidelity)
        sibling_temperature: temperature for contrastive and prototype losses
        detection_weights: (2,) class weight tensor for detection CE

    Returns:
        total_loss, dict of individual loss values
    """
    losses = {}
    device = z.device

    # -- Stage 1: Detection (all samples) ----------------------------------
    det_logits = model.detect(z)
    l_detect = detection_loss(det_logits, y_detect, class_weights=detection_weights)
    losses["detection"] = l_detect.item()
    total = l_detect

    # -- Stage 2: Category (faulty samples only) ---------------------------
    faulty_mask = y_detect == 1
    if faulty_mask.sum() > 0:
        z_faulty = z[faulty_mask]
        y_cat_faulty = y_category[faulty_mask]
        cat_logits = model.categorize(z_faulty)
        l_cat = category_loss(cat_logits, y_cat_faulty)
        losses["category"] = l_cat.item()
        total = total + alpha * l_cat
    else:
        losses["category"] = 0.0

    # -- Stage 3: Root-cause losses (per category) -------------------------
    h_flat = h_groups.reshape(h_groups.shape[0], -1)

    # Accumulators (connected to graph via first addition, no requires_grad needed)
    l_rc_total = torch.tensor(0.0, device=device)
    l_sib_total = torch.tensor(0.0, device=device)
    l_proto_total = torch.tensor(0.0, device=device)
    n_rc_cats = 0

    for cat_idx, cat_name in enumerate(category_names):
        cat_mask = faulty_mask & (y_category == cat_idx)
        if cat_mask.sum() < 1:
            continue

        z_cat = z[cat_mask]
        h_cat_flat = h_flat[cat_mask]
        h_cat_groups = h_groups[cat_mask]
        y_rc_global = y_rootcause[cat_mask]

        if cat_name not in rootcause_local_labels:
            continue
        local_map = rootcause_local_labels[cat_name]
        valid = torch.tensor(
            [int(y_rc_global[i].item()) in local_map for i in range(len(y_rc_global))],
            dtype=torch.bool, device=device
        )
        if valid.sum() < 1:
            continue

        z_valid = z_cat[valid]
        h_valid_flat = h_cat_flat[valid]
        h_valid_groups = h_cat_groups[valid]
        y_rc_local = torch.tensor(
            [local_map[int(y_rc_global[i].item())]
             for i in range(len(y_rc_global)) if int(y_rc_global[i].item()) in local_map],
            dtype=torch.long, device=device
        )

        n_rc_classes = model.category_sizes.get(cat_name, 0)

        # Root-cause CE loss (works with 1+ samples)
        rc_logits = model.diagnose(z_valid, cat_name)
        if rc_logits is not None:
            l_rc = rootcause_loss(rc_logits, y_rc_local)
            l_rc_total = l_rc_total + l_rc

        # Prototype loss on h_groups (trains explanation space directly)
        if gamma > 0 and n_rc_classes >= 2 and len(y_rc_local.unique()) >= 2:
            l_proto = prototype_loss(h_valid_groups, y_rc_local,
                                     n_rc_classes, temperature=sibling_temperature)
            l_proto_total = l_proto_total + l_proto

        # Intra-family contrastive loss (requires 2+ samples with 2+ classes)
        if beta > 0 and valid.sum() >= 2 and len(y_rc_local.unique()) >= 2:
            l_sib = sibling_separation_loss(h_valid_flat, y_rc_local,
                                            temperature=sibling_temperature)
            l_sib_total = l_sib_total + l_sib

        n_rc_cats += 1

    if n_rc_cats > 0:
        l_rc_total = l_rc_total / n_rc_cats
        l_sib_total = l_sib_total / n_rc_cats
        l_proto_total = l_proto_total / n_rc_cats

    losses["rootcause"] = l_rc_total.item()
    losses["sibling"] = l_sib_total.item()
    losses["prototype"] = l_proto_total.item()

    total = total + lambda_ * l_rc_total + beta * l_sib_total + gamma * l_proto_total
    losses["total"] = total.item()

    return total, losses
