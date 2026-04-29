"""Training objective for hierarchical fault diagnosis.

The full loss is

    L = L_detect + alpha * L_cat + lambda_rc * L_rc + L_sep
    L_sep = beta * L_ctr + gamma * L_pm

with the four components:

    L_detect   binary cross-entropy for clean vs faulty (all samples)
    L_cat      class-weighted cross-entropy over fault categories
               (faulty samples only)
    L_rc       per-category cross-entropy over root causes within the
               true fault category
    L_sep      root-cause separation loss applied only to faulty samples
               within their ground-truth fault category. It combines:
                 L_ctr  supervised contrastive term over the flattened
                        group-embedding matrix vec(H), pulling samples
                        with the same root cause together and pushing
                        same-category but different-root-cause samples
                        apart
                 L_pm   prototype-matching cross-entropy that converts
                        squared distances to class prototypes in the
                        group-embedding space into class logits

The contrastive term operates on flattened group embeddings so that the
prototype matcher and the contrastive geometry share the same space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def detection_loss(logits, y_binary, class_weights=None):
    """Binary cross-entropy for clean (0) vs faulty (1)."""
    return F.cross_entropy(logits, y_binary, weight=class_weights)


def category_loss(logits, y_category):
    """Cross-entropy over fault categories, faulty samples only.

    Uses ignore_index=-1 so any clean sample that leaks through the
    caller's filter is skipped instead of causing an out-of-range error.
    """
    return F.cross_entropy(logits, y_category, ignore_index=-1)


def rootcause_loss(logits, y_rootcause):
    """Cross-entropy over root causes within one category."""
    return F.cross_entropy(logits, y_rootcause)


def prototype_matching_loss(h_groups, y_rc_local, n_rc_classes, temperature=0.1):
    """Prototype-matching cross-entropy in the group-embedding space.

    Computes per-class prototypes as the mean group-embedding tensor
    (G x h) over samples sharing the same root-cause label, converts the
    squared Euclidean distance to each prototype into a logit via
    -d / temperature, and applies cross-entropy.

    Args:
        h_groups:      (N, G, h) group embeddings.
        y_rc_local:    (N,) local root-cause labels in {0, ..., n_rc_classes-1}.
        n_rc_classes:  number of root causes in the current category.
        temperature:   scaling factor that converts squared distance to logits.
    """
    n = h_groups.shape[0]
    if n < 2 or len(y_rc_local.unique()) < 2:
        return (h_groups * 0).sum()  # zero loss connected to the graph

    # Per-class prototypes: mean group-embedding tensor.
    protos = []
    for c in range(n_rc_classes):
        mask = y_rc_local == c
        if mask.sum() > 0:
            protos.append(h_groups[mask].mean(dim=0))
        else:
            protos.append(torch.zeros_like(h_groups[0]))
    protos = torch.stack(protos, dim=0)              # (n_rc, G, h)

    # Per-group squared Euclidean distance: (N, n_rc, G).
    diff = h_groups.unsqueeze(1) - protos.unsqueeze(0)
    group_dists = (diff ** 2).sum(dim=-1)

    # Sum across groups for the total distance, convert to logits.
    distances = group_dists.sum(dim=-1)              # (N, n_rc)
    logits = -distances / max(temperature, 0.01)

    return F.cross_entropy(logits, y_rc_local)


def contrastive_separation_loss(h_groups_flat, y_rootcause, temperature=0.1):
    """Supervised contrastive loss on flattened group embeddings.

    Operates on vec(H) for each sample (the row-major flattening of the
    G x h group-embedding matrix). Positive pairs share the same root
    cause within the same fault category; negative pairs share the same
    category but differ in root cause.

    Args:
        h_groups_flat: (N, G*h) flattened group embeddings of samples
                       drawn from one fault category.
        y_rootcause:   (N,) local root-cause labels.
        temperature:   softmax temperature for cosine similarity.
    """
    n = h_groups_flat.shape[0]
    if n < 2:
        return (h_groups_flat * 0).sum()

    unique_labels = y_rootcause.unique()
    if len(unique_labels) < 2:
        return (h_groups_flat * 0).sum()

    z_norm = F.normalize(h_groups_flat, dim=1)
    sim = torch.mm(z_norm, z_norm.t()) / temperature

    labels_eq = y_rootcause.unsqueeze(0) == y_rootcause.unsqueeze(1)
    self_mask = torch.eye(n, dtype=torch.bool, device=h_groups_flat.device)
    pos_mask = labels_eq & ~self_mask

    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim)
    denom = (exp_sim * (~self_mask).float()).sum(dim=1, keepdim=True).clamp(min=1e-8)
    log_prob = sim - torch.log(denom)

    n_pos = pos_mask.float().sum(dim=1)
    has_pos = n_pos > 0
    if has_pos.sum() == 0:
        return (h_groups_flat * 0).sum()

    loss_per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos.clamp(min=1)
    return loss_per_anchor[has_pos].mean()


def compute_detection_weights(y_detect):
    """Compute inverse-frequency class weights for the detection head."""
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
                      alpha=1.0, lambda_rc=1.0, beta=0.5, gamma=0.3,
                      temperature=0.1,
                      detection_weights=None):
    """Combined hierarchical loss.

        L = L_detect + alpha * L_cat + lambda_rc * L_rc + L_sep
        L_sep = beta * L_ctr + gamma * L_pm

    Args:
        model:                  HierarchicalDiagnosisModel.
        z:                      (N, D) projected embeddings.
        h_groups:               (N, G, h) group-level embeddings.
        y_detect:               (N,) binary labels (0 = clean, 1 = faulty).
        y_category:             (N,) category indices, -1 for clean samples.
        y_rootcause:            (N,) global root-cause indices, -1 for
                                clean / invalid.
        category_names:         list of category name strings.
        rootcause_local_labels: dict category_name -> {global_rc: local_idx}.
        alpha:                  weight on the category cross-entropy term.
        lambda_rc:              weight on the root-cause cross-entropy term.
        beta:                   weight on the contrastive term inside L_sep.
        gamma:                  weight on the prototype-matching term inside L_sep.
        temperature:            shared temperature for contrastive + prototype.
        detection_weights:      (2,) class weight tensor for the detection head.

    Returns:
        total_loss, dict of individual loss values keyed by name:
            "detection", "category", "rootcause",
            "contrastive", "prototype",
            "separation" (= beta * L_ctr + gamma * L_pm),
            "total".
    """
    losses = {}
    device = z.device

    # ── L_detect (all samples) ─────────────────────────────────────────────
    det_logits = model.detect(z)
    l_detect = detection_loss(det_logits, y_detect, class_weights=detection_weights)
    losses["detection"] = l_detect.item()
    total = l_detect

    # ── L_cat (faulty samples only) ────────────────────────────────────────
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

    # ── L_rc, L_ctr, L_pm (per category, faulty samples within the
    # ground-truth category only) ──────────────────────────────────────────
    h_flat = h_groups.reshape(h_groups.shape[0], -1)

    l_rc_total = torch.tensor(0.0, device=device)
    l_ctr_total = torch.tensor(0.0, device=device)
    l_pm_total = torch.tensor(0.0, device=device)
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
            dtype=torch.bool, device=device,
        )
        if valid.sum() < 1:
            continue

        z_valid = z_cat[valid]
        h_valid_flat = h_cat_flat[valid]
        h_valid_groups = h_cat_groups[valid]
        y_rc_local = torch.tensor(
            [local_map[int(y_rc_global[i].item())]
             for i in range(len(y_rc_global)) if int(y_rc_global[i].item()) in local_map],
            dtype=torch.long, device=device,
        )

        n_rc_classes = model.category_sizes.get(cat_name, 0)

        # L_rc: per-category cross-entropy.
        rc_logits = model.diagnose(z_valid, cat_name)
        if rc_logits is not None:
            l_rc_total = l_rc_total + rootcause_loss(rc_logits, y_rc_local)

        # L_pm: prototype-matching loss (operates on H).
        if gamma > 0 and n_rc_classes >= 2 and len(y_rc_local.unique()) >= 2:
            l_pm_total = l_pm_total + prototype_matching_loss(
                h_valid_groups, y_rc_local, n_rc_classes, temperature=temperature)

        # L_ctr: supervised contrastive term on vec(H).
        if beta > 0 and valid.sum() >= 2 and len(y_rc_local.unique()) >= 2:
            l_ctr_total = l_ctr_total + contrastive_separation_loss(
                h_valid_flat, y_rc_local, temperature=temperature)

        n_rc_cats += 1

    if n_rc_cats > 0:
        l_rc_total = l_rc_total / n_rc_cats
        l_ctr_total = l_ctr_total / n_rc_cats
        l_pm_total = l_pm_total / n_rc_cats

    # L_sep = beta * L_ctr + gamma * L_pm.
    l_sep = beta * l_ctr_total + gamma * l_pm_total

    losses["rootcause"] = l_rc_total.item()
    losses["contrastive"] = l_ctr_total.item()
    losses["prototype"] = l_pm_total.item()
    losses["separation"] = float(l_sep.detach().item())

    total = total + lambda_rc * l_rc_total + l_sep
    losses["total"] = total.item()

    return total, losses
