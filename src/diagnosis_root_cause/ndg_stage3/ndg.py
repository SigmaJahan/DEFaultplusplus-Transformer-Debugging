from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .mapping import resolve_subsystem, parse_feature_core_map_md, SUBSYSTEMS, display_name, resolve_to_core

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def now_utc_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

@dataclass
class Node:
    id: str
    type: str
    key: str
    attrs: Dict[str, Any]

@dataclass
class Edge:
    type: str
    source: str
    target: str
    attrs: Dict[str, Any]

class NDG:
    def __init__(self, name: str, architecture: str):
        self.name = name
        self.architecture = architecture
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self._counter = 0
        self._index: Dict[Tuple[str,str], str] = {}

    def _new_id(self) -> str:
        self._counter += 1
        return f"n{self._counter:06d}"

    def add_node(self, type_: str, key: str, **attrs) -> str:
        idx = (type_, key)
        if idx in self._index:
            node_id = self._index[idx]
            # merge attrs
            for n in self.nodes:
                if n.id == node_id:
                    for k,v in attrs.items():
                        if v is not None:
                            n.attrs[k] = v
                    break
            return node_id
        node_id = self._new_id()
        self.nodes.append(Node(node_id, type_, key, dict(attrs)))
        self._index[idx] = node_id
        return node_id

    def add_edge(self, type_: str, src: str, tgt: str, **attrs) -> None:
        self.edges.append(Edge(type_, src, tgt, dict(attrs)))

    def to_json(self, provenance: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "created_at_utc": now_utc_iso(),
            "provenance": provenance,
            "meta": meta,
            "nodes": [{"id": n.id, "type": n.type, "key": n.key, **n.attrs} for n in self.nodes],
            "edges": [{"type": e.type, "from": e.source, "to": e.target, **e.attrs} for e in self.edges],
        }

def build_schema(input_files: List[str]) -> Dict[str, Any]:
    return {
        "name": "Neural Diagnosis Graph (NDG)",
        "version": "3.0",
        "created_at_utc": now_utc_iso(),
        "built_from": input_files,
        "definition": (
            "A Neural Diagnosis Graph (NDG) is a directed, typed graph G=(V,E) where nodes represent "
            "model subsystems, diagnostic metrics (core features), and fault hypotheses, and edges encode "
            "evidence-supported causal relations learned from mutation-grounded differential signatures "
            "and explanation artifacts (SHAP/rules/counterfactuals)."
        ),
        "node_types": {
            "XAIReport": "Global explanation summary (SHAP/rules/counterfactuals).",
            "CoreFeature": "Core indicator aggregated across variants.",
            "Subsystem": "Functional subsystem (FFN, LayerNorm, KV-cache, runtime, etc.).",
            "FaultFamily": "Top-level fault category (label space).",
            "FaultSubcategory": "Fine-grained root-cause label within a family (optional).",
            "Remediation": "Practitioner action/check (optional hook).",
        },
        "edge_types": {
            "BELONGS_TO": "CoreFeature -> Subsystem",
            "HIGHLIGHTS": "XAIReport -> CoreFeature (SHAP importance)",
            "CHANGES": "XAIReport -> CoreFeature (counterfactual change count)",
            "ENCODES_RULE": "XAIReport -> FaultFamily (stable rule, fold support)",
            "CONFUSABLE_WITH": "FaultFamily -> FaultFamily (confusion-derived alternatives)",
            "IMPACTS": "FaultFamily -> Subsystem (mutation-grounded impact weight)",
            "SIGNATURE": "FaultFamily -> CoreFeature (mutation-grounded signature; direction/effect)",
        },
        "notes": [
            "Subsystem mapping is loaded from feature_core_map.md when provided. Otherwise a conservative token heuristic is used.",
            "SIGNATURE/IMPACTS edges require diagnosis JSONs (enc_diagnosis.json / dec_diagnosis.json).",
        ],
    }

def _best_experiment_name(experiments: Dict[str, Any]) -> str:
    for k in experiments.keys():
        if k.lower().startswith("xgboost") or k.lower().startswith("xgb"):
            return k
    best_k, best_v = None, -1.0
    for k,v in experiments.items():
        m = safe_get(v, "metrics", default={}) or {}
        score = m.get("macro_f1", m.get("f1_macro", m.get("auroc", 0.0))) or 0.0
        if score > best_v:
            best_v, best_k = score, k
    return best_k or list(experiments.keys())[0]

def summarize_stage12(det: Dict[str, Any], cat: Dict[str, Any], xai: Dict[str, Any], arch: str) -> Dict[str, Any]:
    det_best = _best_experiment_name(det.get("experiments", {}))
    cat_best = _best_experiment_name(cat.get("experiments", {}))
    det_m = safe_get(det, "experiments", det_best, "metrics", default={}) or {}
    cat_m = safe_get(cat, "experiments", cat_best, "metrics", default={}) or {}
    return {
        "architecture": arch,
        "detection": {
            "n_samples": det.get("n_samples"),
            "n_features": det.get("n_features"),
            "class_dist": det.get("class_dist"),
            "cv_note": det.get("cv_note"),
            "best_model": det_best,
            "metrics": {k: det_m.get(k) for k in ["auroc","auprc","f1_macro","f1_weighted","base_rate"] if k in det_m},
        },
        "categorization": {
            "n_samples": cat.get("n_samples"),
            "n_features": cat.get("n_features"),
            "class_dist": cat.get("class_dist"),
            "cv_note": cat.get("cv_note"),
            "best_model": cat_best,
            "labels": cat.get("label_names"),
            "metrics": {k: cat_m.get(k) for k in ["macro_f1","accuracy","auroc","top3_acc","top5_acc","balanced_accuracy"] if k in cat_m},
            "confusion_matrix_normalized": cat_m.get("confusion_matrix_normalized"),
            "confusion_matrix_labels": cat_m.get("confusion_matrix_labels"),
        },
        "xai": {
            "shap_stability_jaccard_top20": safe_get(xai, "shap", "stability_jaccard_top20"),
            "shap_top30_core_features": safe_get(xai, "shap", "top30_core_features", default=[])[:30],
            "rules_mean_fidelity_to_xgb": safe_get(xai, "rules", "mean_fidelity_to_xgb"),
            "stable_rules_across_folds": safe_get(xai, "rules", "stable_rules_across_folds", default=[]),
            "counterfactual_total_generated": safe_get(xai, "counterfactuals", "total_generated"),
            "counterfactual_top_changed_core_features": safe_get(xai, "counterfactuals", "top_changed_core_features", default=[]),
            "layer_pattern": safe_get(xai, "shap", "layer_pattern", default={}),
        },
    }

def add_xai_nodes_edges(g: NDG, xai_sum: Dict[str, Any], labels: List[str],
                        core_map: Optional[Dict[str,str]], core_set: Optional[set] = None) -> None:
    xai_id = g.add_node(
        "XAIReport", f"{g.architecture}_xai",
        shap_stability=xai_sum.get("shap_stability_jaccard_top20"),
        rules_mean_fidelity=xai_sum.get("rules_mean_fidelity_to_xgb"),
        counterfactual_total=xai_sum.get("counterfactual_total_generated"),
        layer_pattern=xai_sum.get("layer_pattern", {}),
    )
    fam_ids = {f: g.add_node("FaultFamily", f) for f in (labels or [])}

    # SHAP core features
    for core, imp in xai_sum.get("shap_top30_core_features", []):
        sub = resolve_subsystem(core, core_map)
        dname = display_name(core, core_set)
        c_id = g.add_node("CoreFeature", core, subsystem=sub, display_name=dname)
        s_id = g.add_node("Subsystem", sub)
        g.add_edge("BELONGS_TO", c_id, s_id)
        g.add_edge("HIGHLIGHTS", xai_id, c_id, importance=float(imp))

    # Counterfactual changes
    for core, cnt in xai_sum.get("counterfactual_top_changed_core_features", []):
        sub = resolve_subsystem(core, core_map)
        dname = display_name(core, core_set)
        c_id = g.add_node("CoreFeature", core, subsystem=sub, display_name=dname)
        s_id = g.add_node("Subsystem", sub)
        g.add_edge("BELONGS_TO", c_id, s_id)
        g.add_edge("CHANGES", xai_id, c_id, count=int(cnt))

    # Rules
    for rule, folds in xai_sum.get("stable_rules_across_folds", []):
        m = re.search(r"=>\s*([A-Za-z0-9_]+)\s*$", rule)
        if not m:
            continue
        cls = m.group(1)
        if cls in fam_ids:
            g.add_edge("ENCODES_RULE", xai_id, fam_ids[cls], rule=rule, support_folds=int(folds))

def add_confusable_edges(g: NDG, labels: List[str], cm_norm: List[List[float]], top_k: int = 2) -> None:
    # top-k off-diagonal per true label
    fam_ids = {n.key: n.id for n in g.nodes if n.type == "FaultFamily"}
    if not labels or not cm_norm:
        return
    for i,true_lab in enumerate(labels):
        if true_lab not in fam_ids:
            continue
        row = cm_norm[i]
        pairs = [(j, row[j]) for j in range(len(row)) if j != i]
        pairs.sort(key=lambda x: x[1], reverse=True)
        for j, rate in pairs[:top_k]:
            if rate <= 0: 
                continue
            pred_lab = labels[j]
            if pred_lab not in fam_ids:
                continue
            g.add_edge("CONFUSABLE_WITH", fam_ids[true_lab], fam_ids[pred_lab], rate=float(rate), source="categorization.confusion")

_LEGACY_SUBSYSTEM = {"attention_score": "attention", "attention_pattern": "attention"}

def _normalize_subsystem(name: str) -> str:
    canon = _LEGACY_SUBSYSTEM.get(name, name)
    return canon if canon in SUBSYSTEMS else name

_DIRECTION_NORM = {"increased": "up", "decreased": "down"}
# Cohen's d > 50 is a numerical artifact (baseline std near zero); cap to keep graph values sane
_MAX_EFFECT = 50.0

def _clip_effect(v: float) -> float:
    return max(min(float(v), _MAX_EFFECT), -_MAX_EFFECT)

def add_mutation_signature_edges(g: NDG, diagnosis: Dict[str, Any],
                                 core_map: Optional[Dict[str,str]], core_set: Optional[set] = None) -> None:
    fam_ids = {n.key: n.id for n in g.nodes if n.type == "FaultFamily"}
    diff = diagnosis.get("differential_signatures", {})
    for fam, obj in diff.items():
        if fam not in fam_ids:
            continue
        # subsystem impacts (mean |Cohen's d| per subsystem -- clip extremes)
        impact = obj.get("subsystem_impact", {})
        for subsystem, w in impact.items():
            subsystem = _normalize_subsystem(subsystem)
            s_id = g.add_node("Subsystem", subsystem)
            g.add_edge("IMPACTS", fam_ids[fam], s_id, impact_weight=_clip_effect(w), source="diagnosis.subsystem_impact")
        # signature features
        for item in obj.get("top20_differential_features", [])[:20]:
            feat = item.get("feature")
            if not feat:
                continue
            raw_sub = item.get("subsystem")
            sub = _normalize_subsystem(raw_sub) if raw_sub else resolve_subsystem(feat, core_map)
            raw_dir = item.get("direction", "")
            direction = _DIRECTION_NORM.get(raw_dir, raw_dir)
            dname = display_name(feat, core_set)
            c_id = g.add_node("CoreFeature", feat, subsystem=sub, display_name=dname)
            g.add_edge(
                "SIGNATURE", fam_ids[fam], c_id,
                direction=direction,
                effect_size=_clip_effect(item.get("effect_size", 0.0)),
                source="diagnosis.differential_signatures",
            )

def build_ndg(
    detection_path: Path,
    categorization_path: Path,
    xai_path: Path,
    out_path: Path,
    feature_core_map_path: Optional[Path] = None,
    diagnosis_path: Optional[Path] = None,
    top_confusions: int = 2,
) -> Dict[str, Any]:
    det = load_json(detection_path)
    cat = load_json(categorization_path)
    xai = load_json(xai_path)
    # minimal validation
    if "experiments" not in det:
        raise ValueError(f"{detection_path}: missing 'experiments' key")
    if "experiments" not in cat:
        raise ValueError(f"{categorization_path}: missing 'experiments' key")
    if "shap" not in xai:
        raise ValueError(f"{xai_path}: missing 'shap' key")
    arch = "encoder" if "enc" in detection_path.name else "decoder" if "dec" in detection_path.name else "unknown"
    summary = summarize_stage12(det, cat, xai, arch)

    core_map = parse_feature_core_map_md(feature_core_map_path) if feature_core_map_path else None
    core_set = set(core_map.keys()) if core_map else None

    g = NDG(f"NDG_{arch}", arch)
    add_xai_nodes_edges(g, summary["xai"], summary["categorization"]["labels"] or [], core_map, core_set)
    # confusions
    cm_labels = summary["categorization"]["confusion_matrix_labels"] or summary["categorization"]["labels"] or []
    cm = summary["categorization"]["confusion_matrix_normalized"]
    if cm and cm_labels:
        add_confusable_edges(g, cm_labels, cm, top_k=top_confusions)

    # mutation-grounded signatures
    if diagnosis_path:
        diagnosis = load_json(diagnosis_path)
        add_mutation_signature_edges(g, diagnosis, core_map, core_set)

    out = g.to_json(provenance={
        "inputs": {
            "detection": detection_path.name,
            "categorization": categorization_path.name,
            "xai": xai_path.name,
            **({"feature_core_map": feature_core_map_path.name} if feature_core_map_path else {}),
            **({"diagnosis": diagnosis_path.name} if diagnosis_path else {}),
        }
    }, meta=summary)
    out_path.write_text(json.dumps(out, indent=2))
    return out
