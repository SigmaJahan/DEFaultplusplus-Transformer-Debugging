from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def load_ndg(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def save_node_edge_type_counts(ndg: Dict[str, Any], out_pdf: Path) -> None:
    node_types = {}
    edge_types = {}
    for n in ndg["nodes"]:
        node_types[n["type"]] = node_types.get(n["type"], 0) + 1
    for e in ndg["edges"]:
        edge_types[e["type"]] = edge_types.get(e["type"], 0) + 1

    fig = plt.figure(figsize=(7.5, 3.0))
    ax1 = plt.subplot(1,2,1)
    ax1.bar(list(node_types.keys()), list(node_types.values()))
    ax1.set_title("Node types")
    ax1.tick_params(axis='x', rotation=45)

    ax2 = plt.subplot(1,2,2)
    ax2.bar(list(edge_types.keys()), list(edge_types.values()))
    ax2.set_title("Edge types")
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

def save_top_shap_bar(ndg: Dict[str, Any], out_pdf: Path, top_k: int = 15) -> None:
    core_by_id = {n["id"]: n for n in ndg["nodes"] if n["type"] == "CoreFeature"}
    rows = []
    for e in ndg["edges"]:
        if e["type"] == "HIGHLIGHTS":
            node = core_by_id.get(e["to"], {})
            label = node.get("display_name") or node.get("key", e["to"])
            rows.append((label, float(e.get("importance", 0.0))))
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:top_k]
    if not rows:
        return
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]

    fig = plt.figure(figsize=(7.5, 4.5))
    plt.barh(labels, vals)
    plt.xlabel("Mean |SHAP| Importance")
    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

def save_confusion_edges_table_plot(ndg: Dict[str, Any], out_pdf: Path, top_k: int = 15) -> None:
    # Plot top confusable edges by rate
    fam_by_id = {n["id"]: n for n in ndg["nodes"] if n["type"] == "FaultFamily"}
    rows = []
    for e in ndg["edges"]:
        if e["type"] == "CONFUSABLE_WITH":
            src = fam_by_id.get(e["from"], {}).get("key", e["from"])
            tgt = fam_by_id.get(e["to"], {}).get("key", e["to"])
            rows.append((f"{src} → {tgt}", float(e.get("rate", 0.0))))
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:top_k]
    if not rows:
        return
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]

    fig = plt.figure(figsize=(7.5, 4.0))
    plt.barh(labels, vals)
    plt.xlabel("Confusion rate")
    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

def save_subsystem_impact_heatmap(ndg: Dict[str, Any], out_pdf: Path, top_families: int = 10, top_subsystems: int = 10) -> None:
    # Extract IMPACTS edges
    fam_nodes = [n for n in ndg["nodes"] if n["type"] == "FaultFamily"]
    sub_nodes = [n for n in ndg["nodes"] if n["type"] == "Subsystem"]
    fam_by_id = {n["id"]: n["key"] for n in fam_nodes}
    sub_by_id = {n["id"]: n["key"] for n in sub_nodes}
    impacts = []
    for e in ndg["edges"]:
        if e["type"] == "IMPACTS":
            impacts.append((fam_by_id.get(e["from"], e["from"]), sub_by_id.get(e["to"], e["to"]), float(e.get("impact_weight", 0.0))))
    if not impacts:
        return

    # choose top families/subsystems by total impact
    fam_tot = {}
    sub_tot = {}
    for f,s,w in impacts:
        fam_tot[f] = fam_tot.get(f, 0.0) + abs(w)
        sub_tot[s] = sub_tot.get(s, 0.0) + abs(w)
    fam_list = sorted(fam_tot.keys(), key=lambda k: fam_tot[k], reverse=True)[:top_families]
    sub_list = sorted(sub_tot.keys(), key=lambda k: sub_tot[k], reverse=True)[:top_subsystems]

    mat = np.zeros((len(fam_list), len(sub_list)))
    idx_f = {f:i for i,f in enumerate(fam_list)}
    idx_s = {s:j for j,s in enumerate(sub_list)}
    for f,s,w in impacts:
        if f in idx_f and s in idx_s:
            mat[idx_f[f], idx_s[s]] = w

    # Use log scale when dynamic range exceeds 1000x to handle extreme effect sizes
    vmax = np.abs(mat).max()
    vmin_nz = np.abs(mat[mat != 0]).min() if np.any(mat != 0) else 1.0
    use_log = (vmax / max(vmin_nz, 1e-12)) > 1000
    fig = plt.figure(figsize=(7.5, 3.5))
    if use_log and vmax > 0:
        norm = mcolors.SymLogNorm(linthresh=max(vmin_nz, 1.0), vmin=0, vmax=vmax)
        plt.imshow(mat, aspect="auto", norm=norm)
        plt.colorbar(label="Impact weight (symlog)")
    else:
        plt.imshow(mat, aspect="auto")
        plt.colorbar(label="Impact weight")
    plt.yticks(range(len(fam_list)), fam_list)
    plt.xticks(range(len(sub_list)), sub_list, rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

def build_publication_plots(ndg_path: Path, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ndg = load_ndg(ndg_path)
    save_node_edge_type_counts(ndg, out_dir/f"{prefix}_type_counts.pdf")
    save_top_shap_bar(ndg, out_dir/f"{prefix}_top_shap.pdf")
    save_confusion_edges_table_plot(ndg, out_dir/f"{prefix}_confusable.pdf")
    save_subsystem_impact_heatmap(ndg, out_dir/f"{prefix}_impact_heatmap.pdf")
