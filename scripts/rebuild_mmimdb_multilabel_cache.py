import argparse
import csv
import json
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild MM-IMDb clean multi-label cache from raw JSON.")
    parser.add_argument("--old-feature-dir", default="cache/text_image_features")
    parser.add_argument("--old-metadata-csv", default="cache/text_image_subset_metadata.csv")
    parser.add_argument("--raw-mmimdb-root", default="/home/M11415102/datasets/mmimdb_full_raw/mmimdb")
    parser.add_argument("--output-feature-dir", default="cache/mmimdb_multilabel_clean")
    parser.add_argument("--output-dir", default="results/mmimdb_multilabel_clean_rebuild")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def log_error(output_dir: Path, title: str) -> None:
    with (output_dir / "errors.log").open("a", encoding="utf-8") as f:
        f.write(f"\n[{title}]\n")
        f.write(traceback.format_exc())
        f.write("\n")


def normalize_sample_id(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    s = Path(s).stem
    if s.startswith("tt"):
        s = s[2:]
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return digits.zfill(7)
    return s


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_text_field(payload: Dict) -> str:
    parts: List[str] = []
    plot = payload.get("plot")
    if isinstance(plot, list):
        parts.extend([str(x).strip() for x in plot if str(x).strip()])
    elif isinstance(plot, str) and plot.strip():
        parts.append(plot.strip())
    plot_outline = payload.get("plot outline") or payload.get("plot_outline")
    if isinstance(plot_outline, str) and plot_outline.strip():
        parts.append(plot_outline.strip())
    seen = set()
    deduped = []
    for part in parts:
        if part not in seen:
            deduped.append(part)
            seen.add(part)
    return " ".join(deduped).strip()


def scan_raw_jsons(raw_root: Path) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    dataset_dir = raw_root / "dataset"
    rows = []
    by_id = {}
    for json_path in sorted(dataset_dir.glob("*.json")):
        sample_id = normalize_sample_id(json_path.stem)
        payload = load_json(json_path)
        genres = payload.get("genres", [])
        if not isinstance(genres, list):
            genres = [str(genres)] if genres else []
        genres = [str(g).strip() for g in genres if str(g).strip()]
        text = choose_text_field(payload)
        image_path = dataset_dir / f"{sample_id}.jpeg"
        row = {
            "sample_id": sample_id,
            "json_path": str(json_path),
            "image_path_if_found": str(image_path) if image_path.exists() else "",
            "genres": json.dumps(genres, ensure_ascii=False),
            "num_genres": len(genres),
            "has_text": bool(text),
            "text_length": len(text),
        }
        rows.append(row)
        by_id[sample_id] = {
            "payload": payload,
            "genres": genres,
            "text": text,
            "json_path": str(json_path),
            "image_path": str(image_path) if image_path.exists() else "",
        }
    return pd.DataFrame(rows), by_id


def load_class_mapping(old_feature_dir: Path) -> Tuple[Dict[str, int], List[str]]:
    feature_meta = json.loads((old_feature_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    raw_mapping = feature_meta["label_mapping"]
    mapping = {str(k): int(v) for k, v in raw_mapping.items()}
    class_names = [None] * (max(mapping.values()) + 1)
    for name, idx in mapping.items():
        class_names[idx] = name
    return mapping, class_names


def build_multihot(genres: List[str], mapping: Dict[str, int]) -> Tuple[np.ndarray, List[int], List[str], List[str]]:
    y = np.zeros(max(mapping.values()) + 1, dtype=np.float32)
    mapped_ids = []
    mapped_names = []
    ignored = []
    for genre in genres:
        if genre in mapping:
            idx = int(mapping[genre])
            y[idx] = 1.0
        else:
            ignored.append(genre)
    mapped_ids = np.flatnonzero(y > 0).astype(int).tolist()
    mapped_names = [name for name, idx in sorted(mapping.items(), key=lambda kv: kv[1]) if idx in mapped_ids]
    return y, mapped_ids, mapped_names, ignored


def load_old_cache(old_feature_dir: Path, old_metadata_csv: Path) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame]:
    meta = pd.read_csv(old_metadata_csv, dtype={"sample_id": str, "id": str, "imdb_id": str})
    meta["normalized_sample_id"] = meta["sample_id"].map(normalize_sample_id)
    features = {}
    for split in ("train", "val", "test"):
        features[split] = {
            "image": np.load(old_feature_dir / f"{split}_image.npy", allow_pickle=True).astype(np.float32),
            "text": np.load(old_feature_dir / f"{split}_text.npy", allow_pickle=True).astype(np.float32),
            "label_raw": np.load(old_feature_dir / f"{split}_label.npy", allow_pickle=True),
            "sample_ids": np.load(old_feature_dir / f"{split}_sample_ids.npy", allow_pickle=True).astype(str),
        }
        features[split]["normalized_sample_ids"] = np.array([normalize_sample_id(x) for x in features[split]["sample_ids"]], dtype=object)
    return features, meta


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    old_feature_dir = (repo_root / args.old_feature_dir).resolve()
    old_metadata_csv = (repo_root / args.old_metadata_csv).resolve()
    raw_root = Path(args.raw_mmimdb_root).resolve()
    output_feature_dir = ensure_dir((repo_root / args.output_feature_dir).resolve())
    output_dir = ensure_dir((repo_root / args.output_dir).resolve())
    write_text(output_dir / "errors.log", "")

    try:
        raw_inventory_df, raw_by_id = scan_raw_jsons(raw_root)
        raw_inventory_df.to_csv(output_dir / "raw_json_inventory.csv", index=False)

        mapping, class_names = load_class_mapping(old_feature_dir)
        (output_dir / "class_mapping.json").write_text(
            json.dumps(
                {
                    "label_mapping": mapping,
                    "class_names": class_names,
                    "num_labels": len(class_names),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        features, old_meta = load_old_cache(old_feature_dir, old_metadata_csv)

        rebuild_rows = []
        clean_meta_rows = []
        sample_alignment_rows = []

        total_old_feature_samples = 0
        matched_samples = 0
        usable_samples = 0
        split_labels: Dict[str, List[np.ndarray]] = defaultdict(list)
        split_images: Dict[str, List[np.ndarray]] = defaultdict(list)
        split_texts: Dict[str, List[np.ndarray]] = defaultdict(list)
        split_sample_ids: Dict[str, List[str]] = defaultdict(list)
        positive_counter_by_split: Dict[str, Counter] = defaultdict(Counter)

        metadata_by_id = old_meta.drop_duplicates("normalized_sample_id").set_index("normalized_sample_id", drop=False)

        for split in ("train", "val", "test"):
            split_meta = old_meta[old_meta["split"].astype(str).str.lower() == split].copy()
            split_meta = split_meta.reset_index(drop=False).rename(columns={"index": "metadata_row_index"})
            split_meta_by_id = split_meta.drop_duplicates("normalized_sample_id").set_index("normalized_sample_id", drop=False)

            for idx, old_sample_id in enumerate(features[split]["sample_ids"]):
                total_old_feature_samples += 1
                normalized = normalize_sample_id(old_sample_id)
                raw_entry = raw_by_id.get(normalized)
                meta_match = split_meta_by_id.loc[normalized] if normalized in split_meta_by_id.index else None
                matched_raw = raw_entry is not None
                matched_samples += int(matched_raw)

                old_label_scalar = int(features[split]["label_raw"][idx]) if np.asarray(features[split]["label_raw"]).ndim == 1 else None
                old_num_positive = 1 if old_label_scalar is not None else int(np.asarray(features[split]["label_raw"][idx]).sum())

                if raw_entry is None:
                    status = "missing_raw_json"
                    new_y = None
                    mapped_ids = []
                    mapped_names = []
                    ignored = []
                    num_positive = 0
                else:
                    new_y, mapped_ids, mapped_names, ignored = build_multihot(raw_entry["genres"], mapping)
                    num_positive = int(new_y.sum())
                    if num_positive == 0:
                        status = "zero_selected_labels"
                    else:
                        status = "ok"

                sample_alignment_rows.append(
                    {
                        "split": split,
                        "old_index": idx,
                        "old_sample_id": old_sample_id,
                        "normalized_sample_id": normalized,
                        "matched_raw_json": matched_raw,
                        "num_positive_labels_old": old_num_positive,
                        "num_positive_labels_new": num_positive,
                        "old_label_id": old_label_scalar,
                        "new_label_ids": json.dumps(mapped_ids),
                        "status": status,
                    }
                )

                rebuild_rows.append(
                    {
                        "sample_id": normalized,
                        "raw_genres": json.dumps(raw_entry["genres"], ensure_ascii=False) if raw_entry else "[]",
                        "mapped_label_ids": json.dumps(mapped_ids),
                        "mapped_label_names": json.dumps(mapped_names, ensure_ascii=False),
                        "num_positive_labels": num_positive,
                        "ignored_genres": json.dumps(ignored, ensure_ascii=False),
                        "status": status,
                    }
                )

                if status != "ok":
                    continue

                usable_samples += 1
                split_images[split].append(features[split]["image"][idx])
                split_texts[split].append(features[split]["text"][idx])
                split_labels[split].append(new_y.astype(np.float32))
                split_sample_ids[split].append(normalized)
                for lid in mapped_ids:
                    positive_counter_by_split[split][lid] += 1

                clean_meta_rows.append(
                    {
                        "sample_id": normalized,
                        "split": split,
                        "old_index_in_split": idx,
                        "metadata_row_index": int(meta_match["metadata_row_index"]) if meta_match is not None else -1,
                        "image_path": str(meta_match["image_path"]) if meta_match is not None and "image_path" in meta_match else raw_entry["image_path"],
                        "text": str(meta_match["text"]) if meta_match is not None and "text" in meta_match else raw_entry["text"],
                        "old_primary_label": str(meta_match["label"]) if meta_match is not None and "label" in meta_match else "",
                        "old_primary_label_id": int(meta_match["label_id"]) if meta_match is not None and "label_id" in meta_match and pd.notna(meta_match["label_id"]) else -1,
                        "raw_genres": json.dumps(raw_entry["genres"], ensure_ascii=False),
                        "mapped_label_ids": json.dumps(mapped_ids),
                        "mapped_label_names": json.dumps(mapped_names, ensure_ascii=False),
                        "num_positive_labels": num_positive,
                        "ignored_genres": json.dumps(ignored, ensure_ascii=False),
                    }
                )

        label_rebuild_df = pd.DataFrame(rebuild_rows).drop_duplicates(subset=["sample_id"], keep="first")
        label_rebuild_df.to_csv(output_dir / "label_rebuild_audit.csv", index=False)
        pd.DataFrame(sample_alignment_rows).to_csv(output_dir / "sample_id_alignment.csv", index=False)

        match_ratio = matched_samples / max(total_old_feature_samples, 1)
        can_reuse = match_ratio >= 0.95 and all(len(split_labels[s]) > 0 for s in ("train", "val", "test"))

        mean_positive_labels = float(label_rebuild_df[label_rebuild_df["status"] == "ok"]["num_positive_labels"].mean()) if not label_rebuild_df.empty else 0.0
        exactly_one_ratio = float((label_rebuild_df[label_rebuild_df["status"] == "ok"]["num_positive_labels"] == 1).mean()) if not label_rebuild_df.empty else 0.0
        multi_positive_ratio = float((label_rebuild_df[label_rebuild_df["status"] == "ok"]["num_positive_labels"] > 1).mean()) if not label_rebuild_df.empty else 0.0
        zero_label_samples = int((label_rebuild_df["status"] == "zero_selected_labels").sum())

        if can_reuse:
            for split in ("train", "val", "test"):
                np.save(output_feature_dir / f"{split}_image.npy", np.stack(split_images[split]).astype(np.float32))
                np.save(output_feature_dir / f"{split}_text.npy", np.stack(split_texts[split]).astype(np.float32))
                np.save(output_feature_dir / f"{split}_label.npy", np.stack(split_labels[split]).astype(np.float32))
                np.save(output_feature_dir / f"{split}_sample_ids.npy", np.array(split_sample_ids[split], dtype=object))

            clean_meta_df = pd.DataFrame(clean_meta_rows)
            clean_meta_df.to_csv(output_dir / "clean_subset_metadata.csv", index=False)
            clean_meta_df.to_csv(output_feature_dir / "clean_subset_metadata.csv", index=False)

            clean_feature_metadata = {
                "source_feature_cache": str(old_feature_dir),
                "source_raw_json_root": str(raw_root),
                "label_mapping": mapping,
                "class_names": class_names,
                "num_labels": len(class_names),
                "train_size": len(split_labels["train"]),
                "val_size": len(split_labels["val"]),
                "test_size": len(split_labels["test"]),
                "mean_positive_labels": mean_positive_labels,
                "positive_label_count_per_split": {
                    split: {class_names[label_id]: int(count) for label_id, count in sorted(counter.items())}
                    for split, counter in positive_counter_by_split.items()
                },
                "rebuild_time": datetime.now().isoformat(),
                "warning_if_reused_features_from_primary_subset": (
                    "Features were reused from cache/text_image_features, which likely came from a primary-label sanity subset. "
                    "Labels were rebuilt from raw MM-IMDb JSON, but subset selection bias may remain."
                ),
            }
            (output_feature_dir / "feature_metadata.json").write_text(
                json.dumps(clean_feature_metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (output_dir / "clean_feature_metadata.json").write_text(
                json.dumps(clean_feature_metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            full_plan = "\n".join(
                [
                    "# FULL REEXTRACT PLAN",
                    "",
                    f"- raw image root: `{raw_root / 'dataset'}`",
                    "- raw text field: prefer `plot` list joined with `plot outline` fallback",
                    "- CLIP model: `openai/clip-vit-base-patch16`",
                    "- split strategy: follow current subset sample_ids if possible, otherwise rebuild deterministic subset with saved split list",
                    f"- label mapping: 17-class mapping from `{old_feature_dir / 'feature_metadata.json'}`",
                    "- expected output cache format: train/val/test image,text,label,sample_ids npy + feature_metadata.json",
                    "- estimated sample count: around current 1700 subset samples if reusing same subset; larger if rebuilding full dataset subset",
                    "- command to run next: build a dedicated extractor that reads raw JSON + image path and writes a new clean feature cache from scratch",
                ]
            )
            write_text(output_dir / "FULL_REEXTRACT_PLAN.md", full_plan)

        summary_lines = [
            "# MM-IMDb Multi-label Rebuild Summary",
            "",
            "## Raw JSON",
            f"raw_samples: {len(raw_inventory_df)}",
            f"usable_samples: {usable_samples}",
            f"num_labels: {len(class_names)}",
            f"mean_positive_labels: {mean_positive_labels:.4f}",
            f"multi_positive_ratio: {multi_positive_ratio:.4f}",
            "",
            "## Alignment",
            f"old_feature_samples: {total_old_feature_samples}",
            f"matched_samples: {matched_samples}",
            f"match_ratio: {match_ratio:.4f}",
            f"can_reuse_old_features: {str(can_reuse).lower()}",
            "",
            "## Rebuild Audit",
            f"ratio_exactly_one_positive: {exactly_one_ratio:.4f}",
            f"zero_label_samples: {zero_label_samples}",
            "",
            "## Decision",
        ]

        if mean_positive_labels <= 1.05:
            summary_lines.append("warning: mean_positive_labels is still close to 1.0; raw parsing may be fine, but this 17-label subset may remain very sparse / primary-label-biased.")

        if match_ratio < 0.95:
            summary_lines.extend(
                [
                    "case: C",
                    "decision: current feature cache cannot be safely relabeled; full feature re-extraction is required.",
                ]
            )
        else:
            summary_lines.extend(
                [
                    "case: A_or_B_pending_baseline",
                    "decision: old features can be reused for a repaired clean cache; final judgment depends on clean baseline quality.",
                ]
            )

        write_text(output_dir / "REBUILD_SUMMARY.md", "\n".join(summary_lines) + "\n")
    except Exception:
        log_error(output_dir, "rebuild_mmimdb_multilabel_cache")
        raise


if __name__ == "__main__":
    main()
