from pathlib import Path
from typing import Dict

import json
import numpy as np
import pandas as pd


def ensure_dir(path):
    """
    Create directory if it does not exist and return Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_metadata(metadata_csv):
    """
    Load metadata CSV and normalize sample_id as string when available.
    """
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata_csv not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].astype(str)
    elif "id" in df.columns:
        df["sample_id"] = df["id"].astype(str)
    elif "imdb_id" in df.columns:
        df["sample_id"] = df["imdb_id"].astype(str)

    return df


def _load_npy_if_exists(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required feature file not found: {path}")
    return np.load(path, allow_pickle=True)


def load_split_arrays(feature_dir, split):
    """
    Load cached frozen CLIP features for one split.

    Expected files:
    - {split}_image.npy
    - {split}_text.npy

    Optional files:
    - {split}_label.npy
    - {split}_labels.npy
    - {split}_sample_id.npy
    - {split}_sample_ids.npy
    """
    feature_dir = Path(feature_dir)

    image = _load_npy_if_exists(feature_dir / f"{split}_image.npy")
    text = _load_npy_if_exists(feature_dir / f"{split}_text.npy")

    payload = {
        "image": image.astype(np.float32),
        "text": text.astype(np.float32),
    }

    for label_name in [f"{split}_label.npy", f"{split}_labels.npy", f"{split}_y.npy"]:
        label_path = feature_dir / label_name
        if label_path.exists():
            payload["label"] = np.load(label_path, allow_pickle=True)
            break

    for id_name in [f"{split}_sample_id.npy", f"{split}_sample_ids.npy", f"{split}_ids.npy"]:
        id_path = feature_dir / id_name
        if id_path.exists():
            payload["sample_id"] = np.load(id_path, allow_pickle=True).astype(str)
            break

    return payload


def ordered_split_metadata(metadata_df, feature_dir, split):
    """
    Return metadata rows ordered to match cached feature rows.

    Priority:
    1. If {split}_sample_id.npy exists, align by sample_id.
    2. Else if metadata has split column, use rows where split == split.
    3. Else assume metadata is already ordered and slice by split sizes.
    """
    feature_dir = Path(feature_dir)
    metadata_df = metadata_df.copy()

    if "sample_id" in metadata_df.columns:
        metadata_df["sample_id"] = metadata_df["sample_id"].astype(str)

    id_candidates = [
        feature_dir / f"{split}_sample_id.npy",
        feature_dir / f"{split}_sample_ids.npy",
        feature_dir / f"{split}_ids.npy",
    ]

    for id_path in id_candidates:
        if id_path.exists():
            sample_ids = np.load(id_path, allow_pickle=True).astype(str)
            if "sample_id" not in metadata_df.columns:
                raise ValueError(
                    f"{id_path.name} exists, but metadata has no sample_id/id/imdb_id column."
                )

            indexed = metadata_df.set_index("sample_id", drop=False)
            missing = [sid for sid in sample_ids if sid not in indexed.index]
            if missing:
                raise ValueError(
                    f"{len(missing)} sample ids from {id_path.name} not found in metadata. "
                    f"Examples: {missing[:5]}"
                )

            return indexed.loc[sample_ids].reset_index(drop=True)

    if "split" in metadata_df.columns:
        split_df = metadata_df[metadata_df["split"].astype(str).str.lower() == split.lower()].copy()
        if len(split_df) > 0:
            return split_df.reset_index(drop=True)

    # Fallback: infer row count from feature files.
    image_path = feature_dir / f"{split}_image.npy"
    if not image_path.exists():
        raise FileNotFoundError(f"Cannot infer split metadata; missing {image_path}")

    n = len(np.load(image_path, mmap_mode="r", allow_pickle=True))

    # Conservative fallback for common order train/val/test.
    split_sizes = {}
    for s in ["train", "val", "test"]:
        p = feature_dir / f"{s}_image.npy"
        if p.exists():
            split_sizes[s] = len(np.load(p, mmap_mode="r", allow_pickle=True))

    if all(s in split_sizes for s in ["train", "val", "test"]):
        start = 0
        if split == "val":
            start = split_sizes["train"]
        elif split == "test":
            start = split_sizes["train"] + split_sizes["val"]
        return metadata_df.iloc[start : start + n].reset_index(drop=True)

    return metadata_df.iloc[:n].reset_index(drop=True)
