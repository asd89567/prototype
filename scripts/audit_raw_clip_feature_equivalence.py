import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recover_cached_feature_baseline import (  # noqa: E402
    ensure_dir,
    load_cache,
    mean_bce_loss_np,
    multilabel_metrics,
    predict,
    threshold_search,
    train_linear,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raw CLIP feature equivalence for MM-IMDb clean cache.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def append_error(output_dir: Path, title: str) -> None:
    with (output_dir / "errors.log").open("a", encoding="utf-8") as f:
        f.write(f"\n[{title}]\n")
        f.write(traceback.format_exc())
        f.write("\n")


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def choose_text_field(payload: Dict) -> str:
    parts: List[str] = []
    plot = payload.get("plot")
    if isinstance(plot, list):
        parts.extend([str(x).strip() for x in plot if str(x).strip()])
    elif isinstance(plot, str) and plot.strip():
        parts.append(plot.strip())
    outline = payload.get("plot outline") or payload.get("plot_outline")
    if isinstance(outline, str) and outline.strip():
        parts.append(outline.strip())
    seen = set()
    out = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            out.append(part)
    return " ".join(out).strip()


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def cosine_rows(raw: np.ndarray, cached: np.ndarray) -> np.ndarray:
    return (l2_normalize_np(raw) * l2_normalize_np(cached)).sum(axis=1)


def load_ordered_records(raw_root: Path, metadata_csv: Path, cache_dir: Path, data: Dict[str, Dict[str, np.ndarray]]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    metadata = pd.read_csv(metadata_csv, dtype={"sample_id": str})
    metadata["sample_id"] = metadata["sample_id"].map(normalize_sample_id)
    indexed = metadata.drop_duplicates("sample_id").set_index("sample_id", drop=False)
    dataset_dir = raw_root / "dataset"
    split_records: Dict[str, pd.DataFrame] = {}
    audit_rows: List[Dict] = []
    for split in ("train", "val", "test"):
        sample_ids = [normalize_sample_id(x) for x in data[split]["sample_ids"]]
        rows = []
        for local_idx, sample_id in enumerate(sample_ids):
            meta_exists = sample_id in indexed.index
            meta_row = indexed.loc[sample_id] if meta_exists else None
            raw_json_path = dataset_dir / f"{sample_id}.json"
            raw_image_path = dataset_dir / f"{sample_id}.jpeg"
            metadata_image = Path(str(meta_row["image_path"])) if meta_exists and "image_path" in indexed.columns else raw_image_path
            text = str(meta_row["text"]) if meta_exists and "text" in indexed.columns and pd.notna(meta_row["text"]) else ""
            raw_text = ""
            raw_json_exists = raw_json_path.exists()
            if raw_json_exists:
                with raw_json_path.open("r", encoding="utf-8") as f:
                    raw_text = choose_text_field(json.load(f))
            raw_text_matches_metadata = bool(text.strip() == raw_text.strip()) if text and raw_text else False
            image_path = metadata_image if metadata_image.exists() else raw_image_path
            row = {
                "split": split,
                "local_idx": local_idx,
                "sample_id": sample_id,
                "metadata_exists": bool(meta_exists),
                "raw_json_exists": bool(raw_json_exists),
                "raw_image_exists": bool(raw_image_path.exists() or metadata_image.exists()),
                "metadata_text_exists": bool(text.strip()),
                "raw_text_exists": bool(raw_text.strip()),
                "raw_text_matches_metadata": raw_text_matches_metadata,
                "image_path": str(image_path),
                "text": text,
            }
            rows.append(row)
            audit_rows.append({k: v for k, v in row.items() if k not in {"text"}})
        split_records[split] = pd.DataFrame(rows)
    alignment_df = pd.DataFrame(audit_rows)
    return split_records, alignment_df


@torch.inference_mode()
def extract_features(
    split_records: Dict[str, pd.DataFrame],
    model_name: str,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for split, records in split_records.items():
        image_features: List[np.ndarray] = []
        text_features: List[np.ndarray] = []
        for start in range(0, len(records), batch_size):
            batch = records.iloc[start : start + batch_size]
            images = []
            for path in batch["image_path"].tolist():
                with Image.open(path) as img:
                    images.append(img.convert("RGB").copy())
            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            image_feat = model.get_image_features(**image_inputs)
            image_feat = torch.nn.functional.normalize(image_feat, dim=-1)
            image_features.append(image_feat.cpu().numpy().astype(np.float32))

            texts = [str(x) for x in batch["text"].tolist()]
            text_inputs = processor(text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_feat = model.get_text_features(**text_inputs)
            text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
            text_features.append(text_feat.cpu().numpy().astype(np.float32))

        out[split] = {
            "image": np.concatenate(image_features, axis=0),
            "text": np.concatenate(text_features, axis=0),
        }
    return out


def run_feature_baseline(
    raw_features: Dict[str, Dict[str, np.ndarray]],
    labels: Dict[str, np.ndarray],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    train_args = SimpleNamespace(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    feature_sets = {
        "text_only": {split: raw_features[split]["text"] for split in ("train", "val", "test")},
        "image_only": {split: raw_features[split]["image"] for split in ("train", "val", "test")},
        "concat_image_text": {
            split: np.concatenate([raw_features[split]["image"], raw_features[split]["text"]], axis=1).astype(np.float32)
            for split in ("train", "val", "test")
        },
    }
    rows: List[Dict] = []
    threshold_rows: List[Dict] = []
    for input_type, features in feature_sets.items():
        model, logs = train_linear(features["train"], labels["train"], features["val"], labels["val"], train_args, device)
        val_logits = predict(model, features["val"], device)
        test_logits = predict(model, features["test"], device)
        candidates, best = threshold_search(val_logits, labels["val"])
        best_strategy = best["threshold_strategy"]
        for cand in candidates:
            metrics = multilabel_metrics(test_logits, labels["test"], cand["threshold"])
            rows.append(
                {
                    "input_type": input_type,
                    "seed": args.seed,
                    "threshold_strategy": cand["threshold_strategy"],
                    "selected_by_val": cand["threshold_strategy"] == best_strategy,
                    "val_macro_f1": cand["val_macro_f1"],
                    "test_macro_f1": metrics["macro_f1"],
                    "test_micro_f1": metrics["micro_f1"],
                    "test_sample_f1": metrics["sample_f1"],
                    "test_bce_loss": mean_bce_loss_np(test_logits, labels["test"]),
                    "mean_predicted_positive_labels": metrics["mean_predicted_positive_labels"],
                    "mean_true_positive_labels": metrics["mean_true_positive_labels"],
                    "epochs_ran": len(logs),
                    "device": str(device),
                }
            )
            threshold_rows.append(
                {
                    "input_type": input_type,
                    "seed": args.seed,
                    "threshold_strategy": cand["threshold_strategy"],
                    "threshold": json.dumps(np.asarray(cand["threshold"]).tolist() if isinstance(cand["threshold"], np.ndarray) else cand["threshold"]),
                    "val_macro_f1": cand["val_macro_f1"],
                    "val_micro_f1": cand["val_micro_f1"],
                    "val_sample_f1": cand["val_sample_f1"],
                    "val_bce_loss": cand["val_bce_loss"],
                }
            )
    pd.DataFrame(threshold_rows).to_csv(output_dir / "raw_feature_threshold_selection.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    metadata_csv = (ROOT / args.metadata_csv).resolve() if not Path(args.metadata_csv).is_absolute() else Path(args.metadata_csv).resolve()
    cache_dir = (ROOT / args.cache_dir).resolve() if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir).resolve()
    output_dir = ensure_dir((ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve())
    (output_dir / "errors.log").touch()
    device = resolve_device(args.device)

    try:
        data, _class_names = load_cache(cache_dir)
        labels = {split: data[split]["label"] for split in ("train", "val", "test")}
        split_records, alignment_df = load_ordered_records(raw_root, metadata_csv, cache_dir, data)
        alignment_df.to_csv(output_dir / "raw_alignment_audit.csv", index=False)
        required_ok = bool(
            alignment_df["metadata_exists"].all()
            and alignment_df["raw_json_exists"].all()
            and alignment_df["raw_image_exists"].all()
            and alignment_df["metadata_text_exists"].all()
            and alignment_df["raw_text_exists"].all()
        )
        if not required_ok:
            raise RuntimeError("Raw sample alignment or raw text/image existence check failed.")

        raw_features = extract_features(split_records, args.clip_model, device, args.clip_batch_size)
        for split in ("train", "val", "test"):
            np.save(output_dir / f"raw_clip_{split}_image.npy", raw_features[split]["image"])
            np.save(output_dir / f"raw_clip_{split}_text.npy", raw_features[split]["text"])

        summary_rows: List[Dict] = []
        sample_rows: List[Dict] = []
        for split in ("train", "val", "test"):
            for branch in ("image", "text"):
                cos = cosine_rows(raw_features[split][branch], data[split][branch])
                for sample_id, value in zip(data[split]["sample_ids"], cos):
                    sample_rows.append(
                        {
                            "split": split,
                            "branch": branch,
                            "sample_id": normalize_sample_id(sample_id),
                            "cosine": float(value),
                        }
                    )
                summary_rows.append(
                    {
                        "split": split,
                        "branch": branch,
                        "num_samples": int(len(cos)),
                        "cosine_mean": float(np.mean(cos)),
                        "cosine_median": float(np.median(cos)),
                        "cosine_min": float(np.min(cos)),
                        "cosine_p05": float(np.quantile(cos, 0.05)),
                        "cosine_p95": float(np.quantile(cos, 0.95)),
                        "pass_mean_gt_0_95": bool(np.mean(cos) > 0.95),
                    }
                )
        feature_eq_df = pd.DataFrame(summary_rows)
        feature_eq_df.to_csv(output_dir / "feature_equivalence.csv", index=False)
        pd.DataFrame(sample_rows).to_csv(output_dir / "feature_equivalence_per_sample.csv", index=False)

        baseline_df = run_feature_baseline(raw_features, labels, args, output_dir, device)
        baseline_df.to_csv(output_dir / "raw_feature_baseline_results.csv", index=False)

        image_mean = float(feature_eq_df[feature_eq_df["branch"] == "image"]["cosine_mean"].mean())
        text_mean = float(feature_eq_df[feature_eq_df["branch"] == "text"]["cosine_mean"].mean())
        selected = baseline_df[baseline_df["selected_by_val"]].copy()
        by_input = selected.set_index("input_type")["test_macro_f1"].to_dict()
        baseline_pass = bool(image_mean > 0.95 and text_mean > 0.95 and min(by_input.values()) > 0.45)
        lines = [
            "# Baseline Recovery Summary",
            "",
            f"- raw_root: {raw_root}",
            f"- metadata_csv: {metadata_csv}",
            f"- cache_dir: {cache_dir}",
            f"- clip_model: {args.clip_model}",
            f"- raw_alignment_passed: {required_ok}",
            f"- raw_image_cosine_mean: {image_mean:.6f}",
            f"- raw_text_cosine_mean: {text_mean:.6f}",
            f"- raw_text_only_macro_f1: {by_input.get('text_only', float('nan')):.6f}",
            f"- raw_image_only_macro_f1: {by_input.get('image_only', float('nan')):.6f}",
            f"- raw_concat_macro_f1: {by_input.get('concat_image_text', float('nan')):.6f}",
            f"- baseline_passed: {baseline_pass}",
        ]
        (output_dir / "BASELINE_RECOVERY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        append_error(output_dir, "audit_raw_clip_feature_equivalence")
        raise


if __name__ == "__main__":
    main()
