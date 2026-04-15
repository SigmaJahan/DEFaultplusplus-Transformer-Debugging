"""Data loading utilities for DEFault++ fault trace datasets."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def load_pickle_dataset(pkl_path: str) -> dict:
    """Load a preprocessed pickle dataset from the DEFault++ project."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_csv_dataset(csv_path: str) -> pd.DataFrame:
    """Load raw CSV dataset with all metadata columns."""
    return pd.read_csv(csv_path)


def load_ndg(ndg_path: str) -> dict:
    """Load a Neural Diagnosis Graph JSON file."""
    with open(ndg_path) as f:
        return json.load(f)


def extract_subsystem_adjacency(ndg: dict) -> tuple[list[str], np.ndarray]:
    """Extract subsystem adjacency matrix from NDG IMPACTS/SIGNATURE edges.

    Returns (subsystem_names, adjacency_matrix) where A[i,j]=1 means
    subsystem i has a known propagation relationship with subsystem j.
    """
    nodes = ndg.get("nodes", [])
    edges = ndg.get("edges", [])

    subsystem_ids = {}
    for node in nodes:
        if node.get("type") == "Subsystem":
            subsystem_ids[node["id"]] = node.get("key", node["id"])

    subsystem_names = sorted(subsystem_ids.values())
    name_to_idx = {n: i for i, n in enumerate(subsystem_names)}
    n = len(subsystem_names)
    adj = np.zeros((n, n), dtype=np.float32)

    feature_to_subsystem = {}
    for edge in edges:
        if edge["type"] == "BELONGS_TO":
            src_id = edge.get("from", edge.get("source"))
            tgt_id = edge.get("to", edge.get("target"))
            if tgt_id in subsystem_ids:
                feature_to_subsystem[src_id] = subsystem_ids[tgt_id]

    for edge in edges:
        if edge["type"] == "IMPACTS":
            src_id = edge.get("from", edge.get("source"))
            tgt_id = edge.get("to", edge.get("target"))
            src_name = None
            tgt_name = None
            if src_id in subsystem_ids:
                src_name = subsystem_ids[src_id]
            if tgt_id in subsystem_ids:
                tgt_name = subsystem_ids[tgt_id]
            if src_name and tgt_name and src_name in name_to_idx and tgt_name in name_to_idx:
                adj[name_to_idx[src_name], name_to_idx[tgt_name]] = 1.0

    for edge in edges:
        if edge["type"] == "CONFUSABLE_WITH":
            src_id = edge.get("from", edge.get("source"))
            tgt_id = edge.get("to", edge.get("target"))
            for node in nodes:
                if node["id"] == src_id:
                    src_key = node.get("key", "")
                if node["id"] == tgt_id:
                    tgt_key = node.get("key", "")

    np.fill_diagonal(adj, 1.0)

    if adj.sum() < n:
        for i in range(n):
            for j in range(n):
                if i != j and adj[i, j] == 0 and adj[j, i] == 0:
                    adj[i, j] = 0.0

    return subsystem_names, adj


def build_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build GroupKFold splits identical to the DEFault++ pipeline."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, y, groups))


def prepare_dataset(data_path: str) -> dict:
    """Load dataset and prepare X, y, groups, feature_names, label_names.

    Accepts pickle (.pkl) or CSV (.csv) paths. For pickle, tries loading
    directly; falls back to CSV inference if deserialization fails.
    """
    if data_path.endswith(".csv"):
        return prepare_dataset_from_csv(data_path, direct_csv=True)

    try:
        data = load_pickle_dataset(data_path)
        X = data["X"].astype(np.float32)
        y_labels = data["y"]
        label_names = data.get("label_names", sorted(set(y_labels)))
        label2int = {label: i for i, label in enumerate(label_names)}
        y = np.array([label2int[label] for label in y_labels], dtype=np.int64)
        groups = data.get("cv_groups", None)
        feature_names = data.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
    except (NotImplementedError, Exception):
        return prepare_dataset_from_csv(data_path)

    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_means = np.nanmean(X, axis=0)
        for col in range(X.shape[1]):
            X[nan_mask[:, col], col] = col_means[col]

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "feature_names": list(feature_names),
        "label_names": list(label_names),
        "label2int": label2int,
        "n_classes": len(label_names),
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
    }


def prepare_dataset_from_csv(data_path: str, direct_csv: bool = False,
                             skip_impute: bool = False) -> dict:
    """Load dataset from CSV.

    When direct_csv=True, data_path is the CSV itself.
    Otherwise, infers the CSV path from a pickle path.

    When skip_impute=True, NaN values are NOT filled. Pass this when the
    FeatureProcessor will handle imputation — it needs raw NaN values to
    correctly identify MNAR columns in Step 1.
    """
    if direct_csv:
        csv_path = Path(data_path)
        is_detection = False
    else:
        pkl_name = Path(data_path).stem
        csv_dir = Path(data_path).parent
        if "enc" in pkl_name:
            csv_path = csv_dir / "encoder_v1_killed_binary.csv"
        else:
            csv_path = csv_dir / "decoder_v1_killed_binary.csv"
        is_detection = "detection" in pkl_name

    target_col = "fault_category"
    print(f"  Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    meta_cols = ["Identifier", "arch", "model_name", "dataset_name", "seed",
                 "is_faulty", "fault_category", "fault_subcategory", "layer_idx",
                 "severity_params", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values.astype(np.float32)
    feature_names = feature_cols

    if is_detection:
        y_labels = df["label"].values
    else:
        y_labels = df[target_col].values
        # Filter out 'correct' class for categorization
        mask = y_labels != "correct"
        X = X[mask]
        y_labels = y_labels[mask]
        df = df[mask]

    label_names = sorted(set(y_labels))
    label2int = {label: i for i, label in enumerate(label_names)}
    y = np.array([label2int[label] for label in y_labels], dtype=np.int64)

    groups = (df["model_name"].astype(str) + "__" +
              df["dataset_name"].astype(str) + "__" +
              df["seed"].astype(str)).values

    nan_mask = np.isnan(X)
    if nan_mask.any() and not skip_impute:
        col_means = np.nanmean(X, axis=0)
        for col in range(X.shape[1]):
            X[nan_mask[:, col], col] = col_means[col]

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "feature_names": list(feature_names),
        "label_names": list(label_names),
        "label2int": label2int,
        "n_classes": len(label_names),
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
    }
