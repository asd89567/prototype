import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

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
from train_prototype_direction_lora_mmimdb import (  # noqa: E402
    MISSING_IMAGE,
    MISSING_TEXT,
    LoRAClipClassifier,
    ProtocolDataset,
    aggregate_delta_stats,
    build_protocol_representations,
    build_prototypes,
    collate_protocol_batch,
    encode_image,
    encode_text,
    forward_lora_batch,
    l2_normalize_np,
    load_ordered_metadata,
    make_missing_table,
    positive_weight,
    selected_threshold_to_json,
    set_seed,
    shared_base_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LoRA baseline health for Prototype Direction LoRA.")
    parser.add_argument("--config", default="configs/prototype_direction_lora_seed42.json")
    parser.add_argument("--output-dir", default="results/lora_baseline_health_audit")
    parser.add_argument("--run-low-lr", action="store_true", help="Also run standard_lora_low_lr.")
    parser.add_argument("--only-low-lr-random", action="store_true", help="Replace standard_lora_low_lr with random-classifier low-LR runs.")
    return parser.parse_args()


def append_error(output_dir: Path, title: str) -> None:
    with (output_dir / "errors.log").open("a", encoding="utf-8") as f:
        f.write(f"\n[{title}]\n")
        f.write(traceback.format_exc())
        f.write("\n")


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), eps)
    return (a * b).sum(axis=1) / denom


def load_raw_online_features(recovery_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        out[split] = {
            "image": np.load(recovery_dir / f"raw_clip_{split}_image.npy").astype(np.float32),
            "text": np.load(recovery_dir / f"raw_clip_{split}_text.npy").astype(np.float32),
        }
        out[split]["full"] = l2_normalize_np((out[split]["image"] + out[split]["text"]) / 2.0)
    return out


def add_metric_rows(
    rows: List[Dict],
    protocol: str,
    method: str,
    split: str,
    seed: int,
    logits: np.ndarray,
    y: np.ndarray,
    threshold_strategy: str,
    threshold,
) -> None:
    metrics = multilabel_metrics(logits, y, threshold)
    rows.append(
        {
            "protocol": protocol,
            "method": method,
            "split": split,
            "seed": seed,
            "threshold_strategy": threshold_strategy,
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "sample_f1": metrics["sample_f1"],
            "bce_loss": mean_bce_loss_np(logits, y),
            "mean_predicted_positive_labels": metrics["mean_predicted_positive_labels"],
            "mean_true_positive_labels": metrics["mean_true_positive_labels"],
        }
    )


def train_online_clip_linear(
    protocol: str,
    cfg: Dict,
    online_base: Dict[str, Dict[str, np.ndarray]],
    labels: Dict[str, np.ndarray],
    tables: Dict[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, np.ndarray], List[Dict], List[Dict], List[Dict]]:
    train_args = SimpleNamespace(
        seed=int(cfg["seed"]),
        epochs=int(cfg["epochs"]),
        patience=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["lr_classifier"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    reps = {split: build_protocol_representations(online_base[split], tables[split]) for split in ("train", "val", "test")}
    model, logs = train_linear(reps["train"], labels["train"], reps["val"], labels["val"], train_args, device)
    val_logits = predict(model, reps["val"], device)
    candidates, best = threshold_search(val_logits, labels["val"])

    threshold_rows: List[Dict] = []
    for cand in candidates:
        threshold_rows.append(
            {
                "protocol": protocol,
                "method": "online_clip_linear",
                "seed": int(cfg["seed"]),
                "threshold_strategy": cand["threshold_strategy"],
                "threshold": selected_threshold_to_json(cand["threshold"]),
                "val_macro_f1": cand["val_macro_f1"],
                "val_micro_f1": cand["val_micro_f1"],
                "val_sample_f1": cand["val_sample_f1"],
                "selected_by_val": cand["threshold_strategy"] == best["threshold_strategy"],
            }
        )

    result_rows: List[Dict] = []
    logits_by_split: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        logits = predict(model, reps[split], device)
        logits_by_split[split] = logits
        np.savez_compressed(output_dir / f"logits_{protocol}_online_clip_linear_{split}.npz", logits=logits, labels=labels[split])
        add_metric_rows(result_rows, protocol, "online_clip_linear", split, int(cfg["seed"]), logits, labels[split], best["threshold_strategy"], best["threshold"])

    training_rows = [
        {
            "protocol": protocol,
            "method": "online_clip_linear",
            "epoch": row["epoch"],
            "train_loss": row["train_loss"],
            "train_cls_loss": row["train_loss"],
            "train_proto_dir_loss": 0.0,
            "train_inst_dir_loss": 0.0,
            "val_macro_f1": row["val_macro_f1"],
            "best_threshold_strategy": row["best_threshold_strategy"],
        }
        for row in logs
    ]
    return model, logits_by_split, result_rows, threshold_rows, training_rows


def classifier_state_from_linear(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        "weight": model.net.weight.detach().cpu().clone(),
        "bias": model.net.bias.detach().cpu().clone(),
    }


def copy_classifier_state(model: LoRAClipClassifier, state: Dict[str, torch.Tensor]) -> None:
    model.classifier.weight.data.copy_(state["weight"].to(model.classifier.weight.device))
    model.classifier.bias.data.copy_(state["bias"].to(model.classifier.bias.device))


@torch.inference_mode()
def zero_lora_feature_equivalence(
    cfg: Dict,
    records: Dict[str, pd.DataFrame],
    cached_base: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    device: torch.device,
) -> Tuple[LoRAClipClassifier, CLIPProcessor, pd.DataFrame]:
    set_seed(int(cfg["seed"]))
    model = LoRAClipClassifier(cfg, int(cfg["num_labels"])).to(device)
    processor = CLIPProcessor.from_pretrained(cfg["clip_model"])
    model.eval()
    rows: List[Dict] = []
    for split in ("train", "val", "test"):
        text_features: List[np.ndarray] = []
        image_features: List[np.ndarray] = []
        rec = records[split]
        for start in range(0, len(rec), int(cfg["batch_size"])):
            batch = rec.iloc[start : start + int(cfg["batch_size"])]
            text_features.append(encode_text(model, processor, batch["text"].astype(str).tolist(), device).cpu().numpy().astype(np.float32))
            image_features.append(encode_image(model, processor, batch["image_path"].astype(str).tolist(), device).cpu().numpy().astype(np.float32))
        encoded = {
            "text": np.concatenate(text_features, axis=0),
            "image": np.concatenate(image_features, axis=0),
        }
        for branch in ("text", "image"):
            diff = encoded[branch] - cached_base[split][branch]
            cos = cosine_np(encoded[branch], cached_base[split][branch])
            rows.append(
                {
                    "split": split,
                    "branch": branch,
                    "num_samples": int(len(cos)),
                    "cosine_mean": float(cos.mean()),
                    "cosine_min": float(cos.min()),
                    "cosine_median": float(np.median(cos)),
                    "max_abs_diff": float(np.max(np.abs(diff))),
                    "mean_abs_diff": float(np.mean(np.abs(diff))),
                    "status": "pass" if float(cos.mean()) > 0.999 and float(np.max(np.abs(diff))) < 1e-4 else "check",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "zero_lora_equivalence.csv", index=False)
    return model, processor, df


@torch.inference_mode()
def zero_lora_logits_equivalence(
    zero_model: LoRAClipClassifier,
    processor: CLIPProcessor,
    protocol: str,
    cfg: Dict,
    datasets: Dict[str, ProtocolDataset],
    online_logits: Dict[str, np.ndarray],
    classifier_state: Dict[str, torch.Tensor],
    device: torch.device,
) -> List[Dict]:
    copy_classifier_state(zero_model, classifier_state)
    zero_model.eval()
    rows: List[Dict] = []
    for split in ("train", "val", "test"):
        loader = DataLoader(datasets[split], batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate_protocol_batch)
        logits_all: List[np.ndarray] = []
        for batch in loader:
            logits, _proto, _inst, _stats = forward_lora_batch(zero_model, processor, batch, device, cfg, collect_stats=False)
            logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
        logits = np.concatenate(logits_all, axis=0)
        diff = logits - online_logits[split]
        rows.append(
            {
                "protocol": protocol,
                "split": split,
                "branch": "logits_same_online_classifier",
                "num_samples": int(logits.shape[0]),
                "cosine_mean": np.nan,
                "cosine_min": np.nan,
                "cosine_median": np.nan,
                "max_abs_diff": float(np.max(np.abs(diff))),
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "status": "pass" if float(np.max(np.abs(diff))) < 1e-4 else "check",
            }
        )
    return rows


def train_lora_audit_method(
    protocol: str,
    method: str,
    cfg: Dict,
    datasets: Dict[str, ProtocolDataset],
    labels: Dict[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
    classifier_state: Dict[str, torch.Tensor] = None,
    is_proto: bool = False,
    lr_lora_scale: float = 1.0,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], Dict]:
    set_seed(int(cfg["seed"]))
    model = LoRAClipClassifier(cfg, int(cfg["num_labels"])).to(device)
    processor = CLIPProcessor.from_pretrained(cfg["clip_model"])
    warmstarted_classifier = classifier_state is not None
    if classifier_state is not None:
        copy_classifier_state(model, classifier_state)
    param_summary = model.trainable_param_summary(protocol, method)

    lora_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier."):
            classifier_params.append(param)
        else:
            lora_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(cfg["lr_lora"]) * lr_lora_scale},
            {"params": classifier_params, "lr": float(cfg["lr_classifier"])},
        ],
        weight_decay=float(cfg["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight(labels["train"], device))
    loader = DataLoader(
        datasets["train"],
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=collate_protocol_batch,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )
    training_rows: List[Dict] = []
    best_state = None
    best_score = -1.0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        sums = {"loss": 0.0, "cls": 0.0, "proto": 0.0, "inst": 0.0, "count": 0}
        for batch in loader:
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, proto_loss, inst_loss, _stats = forward_lora_batch(model, processor, batch, device, cfg, collect_stats=False)
            cls_loss = criterion(logits, y)
            loss = cls_loss
            if is_proto:
                loss = loss + float(cfg["lambda_proto_dir"]) * proto_loss + float(cfg["lambda_inst_dir"]) * inst_loss
            loss.backward()
            optimizer.step()
            n = int(len(y))
            sums["loss"] += float(loss.detach().cpu()) * n
            sums["cls"] += float(cls_loss.detach().cpu()) * n
            sums["proto"] += float(proto_loss.detach().cpu()) * n
            sums["inst"] += float(inst_loss.detach().cpu()) * n
            sums["count"] += n
        val_logits, _ = evaluate_lora_audit(model, processor, datasets["val"], cfg, device, collect_stats=False)
        _candidates, best = threshold_search(val_logits, labels["val"])
        score = float(best["val_macro_f1"])
        training_rows.append(
            {
                "protocol": protocol,
                "method": method,
                "epoch": epoch,
                "train_loss": sums["loss"] / max(sums["count"], 1),
                "train_cls_loss": sums["cls"] / max(sums["count"], 1),
                "train_proto_dir_loss": sums["proto"] / max(sums["count"], 1),
                "train_inst_dir_loss": sums["inst"] / max(sums["count"], 1),
                "val_macro_f1": score,
                "best_threshold_strategy": best["threshold_strategy"],
                "lr_lora_scale": lr_lora_scale,
                "warmstarted_classifier": warmstarted_classifier,
            }
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits, _ = evaluate_lora_audit(model, processor, datasets["val"], cfg, device, collect_stats=False)
    candidates, best = threshold_search(val_logits, labels["val"])
    threshold_rows: List[Dict] = []
    for cand in candidates:
        threshold_rows.append(
            {
                "protocol": protocol,
                "method": method,
                "seed": int(cfg["seed"]),
                "threshold_strategy": cand["threshold_strategy"],
                "threshold": selected_threshold_to_json(cand["threshold"]),
                "val_macro_f1": cand["val_macro_f1"],
                "val_micro_f1": cand["val_micro_f1"],
                "val_sample_f1": cand["val_sample_f1"],
                "selected_by_val": cand["threshold_strategy"] == best["threshold_strategy"],
            }
        )

    result_rows: List[Dict] = []
    delta_rows: List[Dict] = []
    for split in ("train", "val", "test"):
        logits, sample_stats = evaluate_lora_audit(model, processor, datasets[split], cfg, device, collect_stats=True)
        np.savez_compressed(output_dir / f"logits_{protocol}_{method}_{split}.npz", logits=logits, labels=labels[split])
        add_metric_rows(result_rows, protocol, method, split, int(cfg["seed"]), logits, labels[split], best["threshold_strategy"], best["threshold"])
        for row in sample_stats:
            row["protocol"] = protocol
            row["method"] = method
            row["split"] = split
        delta_rows.extend(sample_stats)

    del model
    torch.cuda.empty_cache()
    return result_rows, threshold_rows, training_rows, delta_rows, param_summary


@torch.inference_mode()
def evaluate_lora_audit(
    model: LoRAClipClassifier,
    processor: CLIPProcessor,
    dataset: ProtocolDataset,
    cfg: Dict,
    device: torch.device,
    collect_stats: bool,
) -> Tuple[np.ndarray, List[Dict]]:
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate_protocol_batch)
    model.eval()
    logits_all: List[np.ndarray] = []
    stats: List[Dict] = []
    for batch in loader:
        logits, _proto, _inst, batch_stats = forward_lora_batch(model, processor, batch, device, cfg, collect_stats=collect_stats)
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
        stats.extend(batch_stats)
    return np.concatenate(logits_all, axis=0), stats


def import_current_results(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    src = ROOT / "results/prototype_direction_lora_seed42"
    results = pd.read_csv(src / "results.csv")
    thresholds = pd.read_csv(src / "threshold_selection.csv")
    training = pd.read_csv(src / "training_log.csv")
    delta = pd.read_csv(src / "delta_alignment_stats.csv")
    params = pd.read_json(src / "trainable_param_count.json")

    method_map = {
        "missing_only_clip_linear": "cached_missing_only",
        "standard_lora": "standard_lora_current",
        "prototype_direction_lora": "prototype_direction_current",
    }
    for df in (results, thresholds, training, delta, params):
        if "method" in df.columns:
            df["method"] = df["method"].map(method_map).fillna(df["method"])
    return results, thresholds, training, delta, params


def online_vs_cached_equivalence(
    cached_base: Dict[str, Dict[str, np.ndarray]],
    online_base: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for split in ("train", "val", "test"):
        for branch in ("image", "text", "full"):
            diff = online_base[split][branch] - cached_base[split][branch]
            cos = cosine_np(online_base[split][branch], cached_base[split][branch])
            rows.append(
                {
                    "split": split,
                    "branch": branch,
                    "num_samples": int(len(cos)),
                    "cosine_mean": float(cos.mean()),
                    "cosine_min": float(cos.min()),
                    "cosine_median": float(np.median(cos)),
                    "max_abs_diff": float(np.max(np.abs(diff))),
                    "mean_abs_diff": float(np.mean(np.abs(diff))),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "online_vs_cached_feature_equivalence.csv", index=False)
    return df


def aggregate_delta_collapse(delta_sample_df: pd.DataFrame, current_delta_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    if not current_delta_df.empty:
        for _, row in current_delta_df.iterrows():
            rows.append(
                {
                    "protocol": row["protocol"],
                    "method": row["method"],
                    "split": row["split"],
                    "missing_type": row["missing_type"],
                    "delta_norm_mean": row["delta_norm"],
                    "delta_norm_median": np.nan,
                    "direction_loss_weight_mean": row["direction_loss_weight_mean"],
                    "direction_loss_valid_ratio": row["direction_loss_valid_ratio"],
                    "delta_cosine_to_proto": row["delta_cosine_to_proto"],
                    "delta_cosine_to_instance": row["delta_cosine_to_instance"],
                    "delta_mse_to_instance": row["delta_mse_to_instance"],
                    "num_samples": row["num_samples"],
                }
            )
    if not delta_sample_df.empty:
        for keys, sub in delta_sample_df.groupby(["protocol", "method", "split", "missing_type"]):
            protocol, method, split, missing_type = keys
            rows.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "split": split,
                    "missing_type": missing_type,
                    "delta_norm_mean": float(sub["delta_norm"].mean()),
                    "delta_norm_median": float(sub["delta_norm"].median()),
                    "direction_loss_weight_mean": float(sub["direction_loss_weight"].mean()),
                    "direction_loss_valid_ratio": float(sub["direction_loss_valid"].mean()),
                    "delta_cosine_to_proto": float(sub["delta_cosine_to_proto"].mean()),
                    "delta_cosine_to_instance": float(sub["delta_cosine_to_instance"].mean()),
                    "delta_mse_to_instance": float(sub["delta_mse_to_instance"].mean()),
                    "num_samples": int(len(sub)),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "delta_collapse_analysis.csv", index=False)
    return df


def make_comparison(results_df: pd.DataFrame, collapse_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    test = results_df[results_df["split"] == "test"].copy()
    pivot = test.pivot_table(index="protocol", columns="method", values="macro_f1", aggfunc="mean")
    rows: List[Dict] = []
    for protocol in sorted(test["protocol"].unique()):
        row = {"protocol": protocol}
        for method in sorted(test["method"].unique()):
            if method in pivot.columns and protocol in pivot.index:
                row[method] = float(pivot.loc[protocol, method])
        if "standard_lora_warmstart_classifier" in row and "prototype_direction_lora_warmstart_classifier" in row:
            row["prototype_warmstart_delta_vs_standard_warmstart"] = row["prototype_direction_lora_warmstart_classifier"] - row["standard_lora_warmstart_classifier"]
        if "standard_lora_current" in row and "standard_lora_warmstart_classifier" in row:
            row["standard_warmstart_delta_vs_current"] = row["standard_lora_warmstart_classifier"] - row["standard_lora_current"]
        if "cached_missing_only" in row and "online_clip_linear" in row:
            row["online_delta_vs_cached_missing"] = row["online_clip_linear"] - row["cached_missing_only"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "healthy_lora_comparison.csv", index=False)
    return df


def render_summary(
    output_dir: Path,
    results_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    online_eq_df: pd.DataFrame,
    zero_eq_df: pd.DataFrame,
    collapse_df: pd.DataFrame,
    ran_low_lr: bool,
) -> str:
    test = results_df[results_df["split"] == "test"]
    pivot = test.pivot_table(index="protocol", columns="method", values="macro_f1", aggfunc="mean")
    protocols = ["image_missing_70", "text_missing_70", "both_70"]

    online_deltas = []
    current_drops = []
    warmstart_gains = []
    proto_warm_deltas = []
    for protocol in protocols:
        if protocol in pivot.index:
            online_deltas.append(float(pivot.loc[protocol, "online_clip_linear"] - pivot.loc[protocol, "cached_missing_only"]))
            current_drops.append(float(pivot.loc[protocol, "standard_lora_current"] - pivot.loc[protocol, "cached_missing_only"]))
            warmstart_gains.append(float(pivot.loc[protocol, "standard_lora_warmstart_classifier"] - pivot.loc[protocol, "standard_lora_current"]))
            proto_warm_deltas.append(float(pivot.loc[protocol, "prototype_direction_lora_warmstart_classifier"] - pivot.loc[protocol, "standard_lora_warmstart_classifier"]))

    online_ok = bool(np.nanmean(np.abs(online_deltas)) < 0.02)
    zero_text = zero_eq_df[zero_eq_df["branch"] == "text"]
    zero_image = zero_eq_df[zero_eq_df["branch"] == "image"]
    zero_ok = bool(zero_text["cosine_mean"].mean() > 0.999 and zero_image["cosine_mean"].mean() > 0.999)
    warmstart_fix = bool(np.nanmean(warmstart_gains) > 0.005)
    proto_still_wins = bool(sum(x >= 0 for x in proto_warm_deltas) >= 2 and np.nanmean(proto_warm_deltas) > 0.001)

    if not online_ok:
        main_reason = "online encoding path 問題"
    elif not zero_ok:
        main_reason = "zero LoRA 不等價問題"
    elif warmstart_fix:
        main_reason = "classifier random init 問題 / LoRA joint training 不穩"
    elif np.nanmean(current_drops) < -0.005:
        main_reason = "LoRA over-adaptation 問題"
    else:
        main_reason = "mixed cached/online feature space 或 threshold / macro-F1 calibration 問題"

    test_collapse = collapse_df[(collapse_df["split"] == "test") & (collapse_df["method"].str.contains("prototype_direction"))]
    proto_current = test_collapse[test_collapse["method"] == "prototype_direction_current"]
    proto_warm = test_collapse[test_collapse["method"] == "prototype_direction_lora_warmstart_classifier"]
    current_delta_norm = float(proto_current["delta_norm_mean"].mean()) if not proto_current.empty else float("nan")
    warm_delta_norm = float(proto_warm["delta_norm_mean"].mean()) if not proto_warm.empty else float("nan")
    collapse_flag = bool(current_delta_norm < 0.02 or warm_delta_norm < 0.02)
    alignment_gain = float(
        collapse_df[
            (collapse_df["split"] == "test")
            & (collapse_df["method"] == "prototype_direction_lora_warmstart_classifier")
        ]["delta_cosine_to_proto"].mean()
        - collapse_df[
            (collapse_df["split"] == "test")
            & (collapse_df["method"] == "standard_lora_warmstart_classifier")
        ]["delta_cosine_to_proto"].mean()
    )
    mechanism = "harmful update suppression" if collapse_flag else "semantic direction guidance"
    if proto_still_wins and alignment_gain > 0 and not collapse_flag:
        mechanism = "semantic direction guidance"
    elif proto_still_wins and alignment_gain > 0 and collapse_flag:
        mechanism = "semantic alignment + harmful update suppression"

    best_method = "prototype_direction_lora_warmstart_classifier" if proto_still_wins else "暫時不成立，需要 no-shrink / magnitude-aware prototype loss"
    next_action = "補 seeds 43/44" if proto_still_wins else "先做 no-shrink 或 direction+magnitude ablation，再補 seeds"

    lines = [
        "# LoRA Baseline Health Audit",
        "",
        "1. standard_lora 掉很多點的主因是什麼？",
        f"- 判斷：{main_reason}。",
        "- 補充：online feature path 和 zero-LoRA 等價若通過，代表不是 preprocessing/tokenization/LoRA injection 壞掉；問題較可能是 random classifier + joint LoRA training 的 optimization/calibration 不穩，或 LoRA update 過大。",
        "",
        "2. online_clip_linear 是否接近 cached_missing_only？",
    ]
    for protocol in protocols:
        lines.append(
            f"- {protocol}: cached={pivot.loc[protocol, 'cached_missing_only']:.4f}, online={pivot.loc[protocol, 'online_clip_linear']:.4f}, delta={pivot.loc[protocol, 'online_clip_linear'] - pivot.loc[protocol, 'cached_missing_only']:+.4f}"
        )
    worst_protocol = protocols[int(np.nanargmax(np.abs(online_deltas)))]
    lines.append(f"- 差最多：{worst_protocol}。")
    lines.extend(
        [
            "",
            "3. zero_lora 是否等價 frozen CLIP？",
            f"- text cosine mean={zero_text['cosine_mean'].mean():.6f}, max_abs_diff={zero_text['max_abs_diff'].max():.6g}。",
            f"- image cosine mean={zero_image['cosine_mean'].mean():.6f}, max_abs_diff={zero_image['max_abs_diff'].max():.6g}。",
            "",
            "4. warmstart classifier 是否修復 standard_lora？",
        ]
    )
    for protocol in protocols:
        lines.append(
            f"- {protocol}: current={pivot.loc[protocol, 'standard_lora_current']:.4f}, warmstart={pivot.loc[protocol, 'standard_lora_warmstart_classifier']:.4f}, delta={pivot.loc[protocol, 'standard_lora_warmstart_classifier'] - pivot.loc[protocol, 'standard_lora_current']:+.4f}"
        )
    lines.extend(["", "5. warmstart 後 prototype_direction_lora 是否仍贏 standard_lora？"])
    for protocol in protocols:
        lines.append(
            f"- {protocol}: healthy_lora={pivot.loc[protocol, 'standard_lora_warmstart_classifier']:.4f}, prototype_warmstart={pivot.loc[protocol, 'prototype_direction_lora_warmstart_classifier']:.4f}, delta={pivot.loc[protocol, 'prototype_direction_lora_warmstart_classifier'] - pivot.loc[protocol, 'standard_lora_warmstart_classifier']:+.4f}"
        )
    lines.extend(
        [
            "",
            "6. prototype_direction_lora 的提升是什麼？",
            f"- 判斷：{mechanism}。",
            f"- current prototype delta_norm mean={current_delta_norm:.6f}; warmstart prototype delta_norm mean={warm_delta_norm:.6f}; warmstart alignment gain={alignment_gain:+.4f}。",
            "- 若 delta_norm 很小且 direction weight/valid ratio 低，這代表 loss 允許 shrink escape，應標註為 suppression effect，不應宣稱 strong semantic recovery。",
            "",
            "7. 目前最能支撐 thesis 的 method version 是哪一個？",
            f"- {best_method}。",
            "- 若要更乾淨支撐 thesis，下一版建議做 direction+magnitude loss：用 train instance full-vs-missing displacement norm 或 raw class prototype displacement norm 當 target，避免只把 LoRA 壓到 0。",
            "",
            "8. 是否值得補 seeds 43/44？",
            f"- {'是' if proto_still_wins else '否'}。{next_action}。",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (output_dir / "LORA_BASELINE_HEALTH_SUMMARY.md").write_text(summary, encoding="utf-8")
    return main_reason, mechanism, best_method, next_action, summary


def terminal_summary(
    results_df: pd.DataFrame,
    online_eq_df: pd.DataFrame,
    zero_eq_df: pd.DataFrame,
    collapse_df: pd.DataFrame,
    main_reason: str,
    mechanism: str,
    best_method: str,
    next_action: str,
) -> str:
    test = results_df[results_df["split"] == "test"]
    pivot = test.pivot_table(index="protocol", columns="method", values="macro_f1", aggfunc="mean")
    methods = list(pivot.columns)

    def avg(method: str) -> float:
        return float(pivot[method].mean()) if method in methods else float("nan")

    online_cos = float(online_eq_df[online_eq_df["branch"].isin(["text", "image"])]["cosine_mean"].mean())
    zero_text = float(zero_eq_df[zero_eq_df["branch"] == "text"]["cosine_mean"].mean())
    zero_image = float(zero_eq_df[zero_eq_df["branch"] == "image"]["cosine_mean"].mean())
    zero_max = float(zero_eq_df[zero_eq_df["branch"].isin(["text", "image"])]["max_abs_diff"].max())

    proto_warm = collapse_df[(collapse_df["split"] == "test") & (collapse_df["method"] == "prototype_direction_lora_warmstart_classifier")]
    std_warm = collapse_df[(collapse_df["split"] == "test") & (collapse_df["method"] == "standard_lora_warmstart_classifier")]
    delta_norm = float(proto_warm["delta_norm_mean"].mean()) if not proto_warm.empty else float("nan")
    alignment_gain = float(proto_warm["delta_cosine_to_proto"].mean() - std_warm["delta_cosine_to_proto"].mean()) if not proto_warm.empty and not std_warm.empty else float("nan")
    delta_vs_healthy = avg("prototype_direction_lora_warmstart_classifier") - avg("standard_lora_warmstart_classifier")
    delta_vs_missing = avg("prototype_direction_lora_warmstart_classifier") - avg("cached_missing_only")
    low_lr_value = avg("standard_lora_low_lr") if "standard_lora_low_lr" in methods else float("nan")
    collapse_status = "yes" if delta_norm < 0.02 else "no"

    return "\n".join(
        [
            "# LoRA Baseline Health Audit Summary",
            "",
            "## Online feature sanity",
            f"cached_missing_only: {avg('cached_missing_only'):.4f}",
            f"online_clip_linear: {avg('online_clip_linear'):.4f}",
            f"online_vs_cached_cosine: {online_cos:.6f}",
            f"status: {'pass' if online_cos > 0.999 else 'check'}",
            "",
            "## Zero LoRA sanity",
            f"text_cosine: {zero_text:.6f}",
            f"image_cosine: {zero_image:.6f}",
            f"max_abs_diff: {zero_max:.6g}",
            f"status: {'pass' if zero_text > 0.999 and zero_image > 0.999 and zero_max < 1e-4 else 'check'}",
            "",
            "## Standard LoRA health",
            f"standard_lora_current: {avg('standard_lora_current'):.4f}",
            f"standard_lora_warmstart: {avg('standard_lora_warmstart_classifier'):.4f}",
            f"standard_lora_low_lr: {low_lr_value:.4f}",
            f"main_failure_reason: {main_reason}",
            "",
            "## Prototype direction after healthy baseline",
            f"prototype_direction_current: {avg('prototype_direction_current'):.4f}",
            f"prototype_direction_warmstart: {avg('prototype_direction_lora_warmstart_classifier'):.4f}",
            f"delta_vs_healthy_lora: {delta_vs_healthy:+.4f}",
            f"delta_vs_missing_only: {delta_vs_missing:+.4f}",
            "",
            "## Mechanism",
            f"delta_norm_collapse: {collapse_status}",
            f"alignment_gain: {alignment_gain:+.4f}",
            f"suppression_or_recovery: {mechanism}",
            "",
            "## Decision",
            f"best_method_to_keep: {best_method}",
            f"next_action: {next_action}",
        ]
    )


def main() -> None:
    args = parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    output_dir = ensure_dir((ROOT / args.output_dir).resolve())
    (output_dir / "errors.log").write_text("", encoding="utf-8")
    device = resolve_device()
    set_seed(int(cfg["seed"]))

    try:
        data, class_names = load_cache((ROOT / cfg["feature_dir"]).resolve())
        labels = {split: data[split]["label"].astype(np.float32) for split in ("train", "val", "test")}
        records = load_ordered_metadata((ROOT / cfg["metadata_csv"]).resolve(), data)
        cached_base = shared_base_features(data)
        online_base = load_raw_online_features(ROOT / "results/baseline_recovery")
        if args.only_low_lr_random:
            existing_results = pd.read_csv(output_dir / "baseline_health_results.csv")
            existing_thresholds = pd.read_csv(output_dir / "threshold_selection.csv")
            existing_training = pd.read_csv(output_dir / "training_log.csv")
            existing_collapse = pd.read_csv(output_dir / "delta_collapse_analysis.csv")
            existing_params = json.loads((output_dir / "trainable_param_count.json").read_text(encoding="utf-8"))
            existing_results = existing_results[existing_results["method"] != "standard_lora_low_lr"].copy()
            existing_thresholds = existing_thresholds[existing_thresholds["method"] != "standard_lora_low_lr"].copy()
            existing_training = existing_training[existing_training["method"] != "standard_lora_low_lr"].copy()
            existing_collapse = existing_collapse[existing_collapse["method"] != "standard_lora_low_lr"].copy()
            existing_params = [row for row in existing_params if row.get("method") != "standard_lora_low_lr"]

            _class_deltas, sample_proto_targets, _prototype_stats = build_prototypes(cached_base, labels, class_names)
            low_result_rows: List[Dict] = []
            low_threshold_rows: List[Dict] = []
            low_training_rows: List[Dict] = []
            low_delta_rows: List[Dict] = []
            low_param_rows: List[Dict] = []
            for protocol in cfg["protocols"]:
                tables = {
                    split: make_missing_table(labels[split], protocol, float(cfg["missing_ratio"]), int(cfg["seed"]) + offset)
                    for split, offset in [("train", 0), ("val", 1000), ("test", 2000)]
                }
                datasets = {
                    split: ProtocolDataset(split, records[split], labels[split], cached_base[split], tables[split], sample_proto_targets[split])
                    for split in ("train", "val", "test")
                }
                rows, th_rows, tr_rows, delta_rows, params = train_lora_audit_method(
                    protocol,
                    "standard_lora_low_lr",
                    cfg,
                    datasets,
                    labels,
                    output_dir,
                    device,
                    classifier_state=None,
                    is_proto=False,
                    lr_lora_scale=0.1,
                )
                low_result_rows.extend(rows)
                low_threshold_rows.extend(th_rows)
                low_training_rows.extend(tr_rows)
                low_delta_rows.extend(delta_rows)
                low_param_rows.append(params)

            low_results_df = pd.DataFrame(low_result_rows)
            low_threshold_df = pd.DataFrame(low_threshold_rows)
            low_training_df = pd.DataFrame(low_training_rows)
            low_delta_df = aggregate_delta_collapse(pd.DataFrame(low_delta_rows), pd.DataFrame(), output_dir)
            results_df = pd.concat([existing_results, low_results_df], ignore_index=True)
            thresholds_df = pd.concat([existing_thresholds, low_threshold_df], ignore_index=True)
            training_df = pd.concat([existing_training, low_training_df], ignore_index=True)
            collapse_df = pd.concat([existing_collapse, low_delta_df], ignore_index=True)

            results_df.to_csv(output_dir / "baseline_health_results.csv", index=False)
            thresholds_df.to_csv(output_dir / "threshold_selection.csv", index=False)
            training_df.to_csv(output_dir / "training_log.csv", index=False)
            collapse_df.to_csv(output_dir / "delta_collapse_analysis.csv", index=False)
            save_json(existing_params + low_param_rows, output_dir / "trainable_param_count.json")

            online_eq_df = pd.read_csv(output_dir / "online_vs_cached_feature_equivalence.csv")
            zero_eq_df = pd.read_csv(output_dir / "zero_lora_equivalence.csv")
            comparison_df = make_comparison(results_df, collapse_df, output_dir)
            main_reason, mechanism, best_method, next_action, _summary = render_summary(
                output_dir,
                results_df,
                comparison_df,
                online_eq_df,
                zero_eq_df,
                collapse_df,
                True,
            )
            print(terminal_summary(results_df, online_eq_df, zero_eq_df, collapse_df, main_reason, mechanism, best_method, next_action))
            return

        online_eq_df = online_vs_cached_equivalence(cached_base, online_base, output_dir)
        _class_deltas, sample_proto_targets, _prototype_stats = build_prototypes(cached_base, labels, class_names)

        current_results, current_thresholds, current_training, current_delta, current_params = import_current_results(output_dir)
        result_rows = current_results.to_dict("records")
        threshold_rows = current_thresholds.to_dict("records")
        training_rows = current_training.to_dict("records")
        param_rows = current_params.to_dict("records")
        new_delta_sample_rows: List[Dict] = []

        zero_model, zero_processor, zero_eq_df = zero_lora_feature_equivalence(cfg, records, cached_base, output_dir, device)

        classifier_states: Dict[str, Dict[str, torch.Tensor]] = {}
        online_logits_by_protocol: Dict[str, Dict[str, np.ndarray]] = {}
        datasets_by_protocol: Dict[str, Dict[str, ProtocolDataset]] = {}
        for protocol in cfg["protocols"]:
            tables = {
                split: make_missing_table(labels[split], protocol, float(cfg["missing_ratio"]), int(cfg["seed"]) + offset)
                for split, offset in [("train", 0), ("val", 1000), ("test", 2000)]
            }
            datasets = {
                split: ProtocolDataset(split, records[split], labels[split], cached_base[split], tables[split], sample_proto_targets[split])
                for split in ("train", "val", "test")
            }
            datasets_by_protocol[protocol] = datasets
            online_model, online_logits, rows, th_rows, tr_rows = train_online_clip_linear(protocol, cfg, online_base, labels, tables, output_dir, device)
            classifier_states[protocol] = classifier_state_from_linear(online_model)
            online_logits_by_protocol[protocol] = online_logits
            result_rows.extend(rows)
            threshold_rows.extend(th_rows)
            training_rows.extend(tr_rows)
            zero_eq_extra = zero_lora_logits_equivalence(
                zero_model,
                zero_processor,
                protocol,
                cfg,
                datasets,
                online_logits,
                classifier_states[protocol],
                device,
            )
            zero_eq_df = pd.concat([zero_eq_df, pd.DataFrame(zero_eq_extra)], ignore_index=True)

        zero_eq_df.to_csv(output_dir / "zero_lora_equivalence.csv", index=False)
        del zero_model
        torch.cuda.empty_cache()

        for protocol in cfg["protocols"]:
            for method, is_proto, lr_scale in [
                ("standard_lora_warmstart_classifier", False, 1.0),
                ("prototype_direction_lora_warmstart_classifier", True, 1.0),
            ]:
                rows, th_rows, tr_rows, delta_rows, params = train_lora_audit_method(
                    protocol,
                    method,
                    cfg,
                    datasets_by_protocol[protocol],
                    labels,
                    output_dir,
                    device,
                    classifier_states[protocol],
                    is_proto=is_proto,
                    lr_lora_scale=lr_scale,
                )
                result_rows.extend(rows)
                threshold_rows.extend(th_rows)
                training_rows.extend(tr_rows)
                new_delta_sample_rows.extend(delta_rows)
                param_rows.append(params)

            if args.run_low_lr:
                rows, th_rows, tr_rows, delta_rows, params = train_lora_audit_method(
                    protocol,
                    "standard_lora_low_lr",
                    cfg,
                    datasets_by_protocol[protocol],
                    labels,
                    output_dir,
                    device,
                    None,
                    is_proto=False,
                    lr_lora_scale=0.1,
                )
                result_rows.extend(rows)
                threshold_rows.extend(th_rows)
                training_rows.extend(tr_rows)
                new_delta_sample_rows.extend(delta_rows)
                param_rows.append(params)

        results_df = pd.DataFrame(result_rows)
        thresholds_df = pd.DataFrame(threshold_rows)
        training_df = pd.DataFrame(training_rows)
        new_delta_sample_df = pd.DataFrame(new_delta_sample_rows)
        collapse_df = aggregate_delta_collapse(new_delta_sample_df, current_delta, output_dir)
        comparison_df = make_comparison(results_df, collapse_df, output_dir)

        results_df.to_csv(output_dir / "baseline_health_results.csv", index=False)
        thresholds_df.to_csv(output_dir / "threshold_selection.csv", index=False)
        training_df.to_csv(output_dir / "training_log.csv", index=False)
        save_json(param_rows, output_dir / "trainable_param_count.json")
        main_reason, mechanism, best_method, next_action, _summary = render_summary(
            output_dir,
            results_df,
            comparison_df,
            online_eq_df,
            zero_eq_df,
            collapse_df,
            args.run_low_lr,
        )
        print(terminal_summary(results_df, online_eq_df, zero_eq_df, collapse_df, main_reason, mechanism, best_method, next_action))
    except Exception:
        append_error(output_dir, "audit_lora_baseline_health")
        raise


if __name__ == "__main__":
    main()
