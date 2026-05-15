import argparse
import copy
import json
import math
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_prototype_conditioned_feature_adapter import (
    MixtureFeatureAdapter,
    RepresentationClassifier,
    StaticFeatureAdapter,
    apply_thresholds,
    concat_full,
    cosine_mean,
    device_from_args,
    ensure_dir,
    load_mmimdb_payload,
    mean_bce_loss,
    multilabel_metrics,
    positive_weight,
    predict_representation,
    resolve_path,
    save_json,
    select_best_thresholds,
    set_seed,
    sigmoid,
    train_representation_classifier,
)


MISSING_COMPLETE = 0
MISSING_TEXT = 1
MISSING_IMAGE = 2

VALID_METHODS = (
    "missing_only",
    "static_feature_adapter",
    "wide_static_feature_adapter",
    "mixture_feature_adapter_unsupervised",
    "safe_mixture_feature_adapter",
    "safe_mixture_beta_zero",
    "safe_mixture_beta_one",
    "safe_mixture_beta_learned",
    "safe_mixture_beta_biased",
    "prototype_plain_hard",
)

SPLIT_OFFSETS = {"train": 0, "val": 10_000, "test": 20_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoRA-aligned feature adapter with matched-oracle and stratified-complete fixes.")
    parser.add_argument("--feature-dir", default="cache/text_image_features")
    parser.add_argument("--metadata-csv", default="cache/text_image_subset_metadata.csv")
    parser.add_argument("--harm-target-csv", default="results/harm_aware_adapter/harm_targets.csv")
    parser.add_argument("--output-dir", default="results/prototype_adapter_fix")
    parser.add_argument("--reference-results-dir", default="results/mora_aligned_feature_adapter")
    parser.add_argument("--dataset", default="mmimdb")
    parser.add_argument("--protocols", nargs="+", default=["image_missing_70"], choices=["image_missing_70", "text_missing_70", "both_70"])
    parser.add_argument("--missing-ratio", type=float, default=0.7)
    parser.add_argument("--both-ratio", type=float, default=0.5)
    parser.add_argument("--complete-sampling", default="random", choices=["random", "stratified_primary_label"])
    parser.add_argument("--min-complete-per-label", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--adapter-rank", type=int, default=16)
    parser.add_argument("--wide-static-rank", type=int, default=64)
    parser.add_argument("--mixture-k", type=int, default=4)
    parser.add_argument("--prototype-k", type=int, default=4)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--lambda-router", type=float, default=0.03)
    parser.add_argument("--lambda-router-list", type=float, nargs="*", default=None)
    parser.add_argument("--oracle-soft-temperatures", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--matched-oracle-eval", action="store_true")
    parser.add_argument("--save-prototype-diagnostics", action="store_true")
    parser.add_argument("--loss-normalize-by-sample-type", action="store_true")
    parser.add_argument(
        "--complete-eval-mode",
        default="branch_ensemble",
        choices=["branch_ensemble", "full_teacher"],
        help="How to evaluate complete samples. branch_ensemble keeps the full teacher training-only.",
    )
    parser.add_argument("--enable-safe-beta", action="store_true", help="Enable beta safe controller")
    parser.add_argument("--beta-bias-image", type=float, default=1.0)
    parser.add_argument("--beta-bias-text", type=float, default=-2.0)
    parser.add_argument("--branch-router-layernorm", action="store_true", help="Branch specific layernorm")
    parser.add_argument("--per-branch-thresholds", action="store_true")
    parser.add_argument("--methods", nargs="+", default=list(VALID_METHODS), choices=list(VALID_METHODS))
    parser.add_argument("--selection-metric", default="macro_f1", choices=["macro_f1"])
    parser.add_argument("--tune-thresholds", action="store_true", default=True)
    parser.add_argument("--threshold-strategies", nargs="+", default=["global", "per_class"], choices=["global", "per_class"])
    parser.add_argument("--threshold-grid", type=float, nargs="+", default=[round(i / 100, 2) for i in range(5, 100, 5)])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def log_error(output_dir: Path, title: str) -> None:
    with (output_dir / "errors.log").open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{title}]\n")
        handle.write(traceback.format_exc())
        handle.write("\n")


def write_csv(rows: List[Dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def load_reference_outputs(reference_dir: Path) -> Dict[str, pd.DataFrame]:
    outputs = {}
    for name in ["mora_aligned_results.csv", "router_prediction_stats.csv", "seed_summary.csv"]:
        path = reference_dir / name
        outputs[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return outputs


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 1:
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def available_branches_for_protocol(protocol: str) -> List[str]:
    if protocol == "image_missing_70":
        return ["image_missing"]
    if protocol == "text_missing_70":
        return ["text_missing"]
    return ["image_missing", "text_missing"]


def branch_feature(split_payload: Dict[str, np.ndarray], branch: str) -> np.ndarray:
    if branch == "image_missing":
        return split_payload["text"].astype(np.float32)
    if branch == "text_missing":
        return split_payload["image"].astype(np.float32)
    raise ValueError(branch)


def branch_train_mask(table: np.ndarray, branch: str) -> np.ndarray:
    if branch == "image_missing":
        return (table == MISSING_COMPLETE) | (table == MISSING_IMAGE)
    if branch == "text_missing":
        return (table == MISSING_COMPLETE) | (table == MISSING_TEXT)
    raise ValueError(branch)


def subgroup_name(value: int) -> str:
    if value == MISSING_COMPLETE:
        return "complete"
    if value == MISSING_IMAGE:
        return "image_missing"
    if value == MISSING_TEXT:
        return "text_missing"
    return "unknown"


def make_loader(*arrays: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    tensors = []
    for array in arrays:
        dtype = torch.long if array.dtype.kind in {"i", "u"} and array.ndim == 1 else torch.float32
        tensors.append(torch.tensor(array, dtype=dtype))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=generator)


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, eps)


def fit_cluster(residual: np.ndarray, k: int, seed: int) -> np.ndarray:
    model = KMeans(n_clusters=k, n_init=20, random_state=seed)
    model.fit(normalize_rows(residual))
    return model.cluster_centers_.astype(np.float32)


def assign_cluster(residual: np.ndarray, centers: np.ndarray) -> np.ndarray:
    sims = normalize_rows(residual) @ normalize_rows(centers).T
    return sims.argmax(axis=1).astype(np.int64)


def soft_cluster_alpha(residual: np.ndarray, centers: np.ndarray, temperature: float) -> np.ndarray:
    scores = (normalize_rows(residual) @ normalize_rows(centers).T) / max(temperature, 1e-6)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return (exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def missing_table_summary(table: np.ndarray, split: str, protocol: str, seed: int, sampling_mode: str) -> Dict:
    n_total = len(table)
    n_complete = int((table == MISSING_COMPLETE).sum())
    n_image = int((table == MISSING_IMAGE).sum())
    n_text = int((table == MISSING_TEXT).sum())
    return {
        "split": split,
        "protocol": protocol,
        "seed": seed,
        "sampling_mode": sampling_mode,
        "n_total": n_total,
        "n_complete": n_complete,
        "n_image_missing": n_image,
        "n_text_missing": n_text,
        "complete_ratio": n_complete / max(n_total, 1),
        "image_missing_ratio": n_image / max(n_total, 1),
        "text_missing_ratio": n_text / max(n_total, 1),
    }


def primary_labels_for_split(split_meta: Dict[str, pd.DataFrame], labels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    primary = {}
    for split, meta in split_meta.items():
        if "label_id" in meta.columns:
            primary[split] = meta["label_id"].to_numpy(dtype=np.int64)
        else:
            y = labels[split]
            primary[split] = y.argmax(axis=1).astype(np.int64)
    return primary


def stratified_complete_indices(
    primary_labels: np.ndarray,
    n_complete: int,
    min_complete_per_label: int,
    seed: int,
) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    unique_labels, counts = np.unique(primary_labels, return_counts=True)
    target_ratio = n_complete / max(len(primary_labels), 1)
    base = {int(label): int(math.floor(count * target_ratio)) for label, count in zip(unique_labels, counts)}
    warnings = []
    feasible_min = min_complete_per_label > 0 and sum(min(int(count), min_complete_per_label) for count in counts) <= n_complete
    if min_complete_per_label > 0 and not feasible_min:
        warnings.append("min_complete_per_label_not_feasible")
    if feasible_min:
        for label, count in zip(unique_labels, counts):
            base[int(label)] = max(base[int(label)], min(int(count), min_complete_per_label))
    capped = {label: min(base[label], int(count)) for label, count in zip(unique_labels, counts)}
    total = sum(capped.values())
    raw = {int(label): count * target_ratio for label, count in zip(unique_labels, counts)}
    if total > n_complete:
        removable = sorted(unique_labels.tolist(), key=lambda label: (capped[int(label)] - raw[int(label)], capped[int(label)]), reverse=True)
        idx = 0
        while total > n_complete and idx < len(removable):
            label = int(removable[idx])
            floor_raw = int(math.floor(raw[label]))
            lower = floor_raw
            if feasible_min:
                lower = max(lower, min_complete_per_label)
            if capped[label] > lower:
                capped[label] -= 1
                total -= 1
            else:
                idx += 1
    if total < n_complete:
        remainders = sorted(
            unique_labels.tolist(),
            key=lambda label: (raw[int(label)] - math.floor(raw[int(label)]), counts[unique_labels.tolist().index(label)] - capped[int(label)]),
            reverse=True,
        )
        idx = 0
        while total < n_complete and idx < len(remainders):
            label = int(remainders[idx])
            label_count = int(counts[unique_labels.tolist().index(label)])
            if capped[label] < label_count:
                capped[label] += 1
                total += 1
            else:
                idx += 1
        if total < n_complete:
            leftovers = [int(label) for label, count in zip(unique_labels, counts) if capped[int(label)] < int(count)]
            for label in leftovers:
                if total >= n_complete:
                    break
                capped[label] += 1
                total += 1
    chosen = []
    for label in unique_labels.tolist():
        label = int(label)
        label_indices = np.flatnonzero(primary_labels == label)
        rng.shuffle(label_indices)
        chosen.extend(label_indices[: capped[label]].tolist())
    chosen = np.array(sorted(chosen), dtype=np.int64)
    if len(chosen) > n_complete:
        chosen = rng.choice(chosen, size=n_complete, replace=False)
        chosen = np.sort(chosen)
        warnings.append("trimmed_complete_indices_to_target")
    elif len(chosen) < n_complete:
        remaining = np.setdiff1d(np.arange(len(primary_labels)), chosen, assume_unique=False)
        extra = rng.choice(remaining, size=n_complete - len(chosen), replace=False)
        chosen = np.sort(np.concatenate([chosen, extra]))
        warnings.append("filled_remaining_complete_indices_randomly")
    return chosen.astype(np.int64), warnings


def make_missing_table(
    n_samples: int,
    protocol: str,
    missing_ratio: float,
    both_ratio: float,
    seed: int,
    split: str,
    sampling_mode: str,
    primary_labels: Optional[np.ndarray],
    min_complete_per_label: int,
) -> Tuple[np.ndarray, List[str]]:
    rng = np.random.default_rng(seed + SPLIT_OFFSETS.get(split, 0))
    n_missing = int(round(n_samples * missing_ratio))
    n_complete = n_samples - n_missing
    if sampling_mode == "stratified_primary_label" and primary_labels is not None:
        complete_indices, warnings = stratified_complete_indices(
            primary_labels,
            n_complete,
            min_complete_per_label if split == "train" else 0,
            seed + SPLIT_OFFSETS.get(split, 0),
        )
    else:
        complete_indices = np.sort(rng.choice(np.arange(n_samples), size=n_complete, replace=False))
        warnings = []
    table = np.zeros(n_samples, dtype=np.int64)
    remaining = np.setdiff1d(np.arange(n_samples), complete_indices, assume_unique=False)
    rng.shuffle(remaining)
    if protocol == "image_missing_70":
        table[remaining] = MISSING_IMAGE
    elif protocol == "text_missing_70":
        table[remaining] = MISSING_TEXT
    elif protocol == "both_70":
        n_image = int(round(len(remaining) * both_ratio))
        table[remaining[:n_image]] = MISSING_IMAGE
        table[remaining[n_image:]] = MISSING_TEXT
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    return table, warnings


def label_distribution_rows(
    protocol: str,
    seed: int,
    split: str,
    sampling_mode: str,
    table: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    min_complete_per_label: int,
) -> List[Dict]:
    complete = table == MISSING_COMPLETE
    overall_counts = y.sum(axis=0)
    complete_counts = y[complete].sum(axis=0) if complete.any() else np.zeros(y.shape[1], dtype=np.float32)
    overall_ratio = overall_counts / max(len(y), 1)
    complete_ratio = complete_counts / max(int(complete.sum()), 1)
    rows = []
    for label_id, label_name in enumerate(class_names):
        rows.append(
            {
                "split": split,
                "seed": seed,
                "protocol": protocol,
                "sampling_mode": sampling_mode,
                "n_total": int(len(table)),
                "n_complete": int(complete.sum()),
                "n_image_missing": int((table == MISSING_IMAGE).sum()),
                "complete_ratio": float(complete.mean()),
                "label_id": label_id,
                "label_name": label_name,
                "overall_label_count": int(overall_counts[label_id]),
                "complete_label_count": int(complete_counts[label_id]),
                "complete_label_ratio": float(complete_ratio[label_id]),
                "overall_label_ratio": float(overall_ratio[label_id]),
                "abs_distribution_gap": float(abs(complete_ratio[label_id] - overall_ratio[label_id])),
                "min_complete_per_label_satisfied": bool(int(complete_counts[label_id]) >= min_complete_per_label if split == "train" and min_complete_per_label > 0 else True),
            }
        )
    return rows


def count_params(module: Optional[nn.Module], trainable_only: bool = True) -> int:
    if module is None:
        return 0
    params = module.parameters()
    if trainable_only:
        return int(sum(p.numel() for p in params if p.requires_grad))
    return int(sum(p.numel() for p in params))


def adapter_param_row(method: str, protocol: str, branch: str, seed: int, model: nn.Module, macro_f1: float, static_params: Optional[int]) -> Dict:
    adapter_params = 0
    router_params = 0
    classifier_params = count_params(getattr(model, "classifier", None), True)
    if hasattr(model, "adapter"):
        adapter_params = count_params(model.adapter, True)
    if hasattr(model, "adapters"):
        adapter_params = sum(count_params(adapter, True) for adapter in model.adapters)
    if hasattr(model, "router"):
        router_params = count_params(model.router, True)
    trainable = count_params(model, True)
    return {
        "protocol": protocol,
        "method": method,
        "branch": branch,
        "seed": seed,
        "trainable_params": trainable,
        "adapter_params": adapter_params,
        "router_params": router_params,
        "classifier_params": classifier_params,
        "macro_f1": macro_f1,
        "params_vs_static_ratio": trainable / static_params if static_params else float("nan"),
    }


@torch.inference_mode()
def predict_branch(model: RepresentationClassifier, x: np.ndarray, device: torch.device) -> Dict[str, np.ndarray]:
    logits, z = predict_representation(model, x, device)
    return {"logits": logits, "z": z}


@torch.inference_mode()
def predict_adapter_mode(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    cluster_override: Optional[np.ndarray] = None,
    alpha_override: Optional[np.ndarray] = None,
    beta_override: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Adapter inference that always goes through model.forward().

    This avoids the previous train/eval mismatch where validation/test manually
    recomputed mixture_delta and silently skipped safe beta + branch LayerNorm.
    """
    model.eval()
    keys = [
        "logits",
        "z_before",
        "z_static",
        "z_after",
        "static_delta",
        "mixture_delta",
        "delta",
        "beta",
        "base_logits",
        "alpha",
    ]
    outputs: Dict[str, List[np.ndarray]] = {key: [] for key in keys}
    router_logits_out: List[np.ndarray] = []
    for start in range(0, len(x), 512):
        end = min(start + 512, len(x))
        batch = torch.tensor(x[start:end], dtype=torch.float32, device=device)
        cluster_tensor = None
        if cluster_override is not None:
            cluster_tensor = torch.tensor(cluster_override[start:end], dtype=torch.long, device=device)
        alpha_tensor = None
        if alpha_override is not None:
            alpha_tensor = torch.tensor(alpha_override[start:end], dtype=torch.float32, device=device)
        beta_tensor = None
        if beta_override is not None:
            beta_tensor = torch.tensor(beta_override[start:end], dtype=torch.float32, device=device)
            if beta_tensor.ndim == 1:
                beta_tensor = beta_tensor.unsqueeze(1)

        if isinstance(model, MixtureFeatureAdapter):
            logits, details = model(
                batch,
                oracle_cluster=cluster_tensor,
                alpha_override=alpha_tensor,
                beta_override=beta_tensor,
                return_details=True,
            )
        else:
            logits, details = model(batch, return_details=True)
            z = details["z_before"]
            details.setdefault("z_static", z)
            details.setdefault("static_delta", torch.zeros_like(z))
            details.setdefault("mixture_delta", details.get("delta", torch.zeros_like(z)))
            details.setdefault("beta", torch.ones((z.size(0), 1), dtype=z.dtype, device=z.device))

        outputs["logits"].append(logits.cpu().numpy())
        for key in keys:
            if key == "logits":
                continue
            outputs[key].append(details[key].cpu().numpy())
        if "router_logits" in details:
            router_logits_out.append(details["router_logits"].cpu().numpy())
    merged = {key: np.concatenate(value, axis=0) for key, value in outputs.items()}
    if router_logits_out:
        merged["router_logits"] = np.concatenate(router_logits_out, axis=0)
    return merged


def per_sample_bce(logits: torch.Tensor, y: torch.Tensor, pos_weight: Optional[torch.Tensor]) -> torch.Tensor:
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    return losses.mean(dim=1)


def train_branch_base(
    model: RepresentationClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    stage: str,
    protocol: str,
    training_rows: List[Dict],
) -> Tuple[RepresentationClassifier, Dict]:
    args.seed = seed
    local_rows: List[Dict] = []
    model, threshold = train_representation_classifier(model, x_train, y_train, x_val, y_val, args, device, stage, local_rows)
    for row in local_rows:
        row["protocol"] = protocol
        row["seed"] = seed
        row["sampling_mode"] = args.complete_sampling
        training_rows.append(row)
    return model, threshold


def train_adapter(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    z_full_train: np.ndarray,
    complete_mask: np.ndarray,
    cluster_labels: Optional[np.ndarray],
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_logits_builder,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    protocol: str,
    method: str,
    branch: str,
    training_rows: List[Dict],
    lambda_align: float,
    lambda_router: float,
) -> Tuple[nn.Module, Dict, List[Dict]]:
    model.to(device)
    pos_weight = None if args.no_pos_weight else positive_weight(y_train, device)
    mse = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    cluster_values = cluster_labels if cluster_labels is not None else np.full(len(x_train), -1, dtype=np.int64)
    loader = make_loader(
        x_train.astype(np.float32),
        y_train.astype(np.float32),
        z_full_train.astype(np.float32),
        complete_mask.astype(np.float32),
        cluster_values.astype(np.int64),
        batch_size=args.batch_size,
        shuffle=True,
        seed=seed,
    )
    best_state = None
    best_score = -1.0
    best_threshold = None
    stale = 0
    weight_stats = {
        "complete": {"n": 0, "cls": 0.0, "align": 0.0, "router": 0.0, "total": 0.0},
        "incomplete": {"n": 0, "cls": 0.0, "align": 0.0, "router": 0.0, "total": 0.0},
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y, batch_z_full, batch_complete, batch_cluster in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_z_full = batch_z_full.to(device)
            batch_complete = batch_complete.to(device) > 0.5
            batch_cluster = batch_cluster.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, details = model(batch_x, return_details=True)
            cls_sample = per_sample_bce(logits, batch_y, pos_weight)
            align_sample = mse(details["z_after"], batch_z_full).mean(dim=1)
            valid = (batch_cluster >= 0) & batch_complete
            
            if args.loss_normalize_by_sample_type:
                cls_loss = 0.0
                n_types_cls = 0
                if batch_complete.any():
                    cls_loss += cls_sample[batch_complete].mean()
                    n_types_cls += 1
                if (~batch_complete).any():
                    cls_loss += cls_sample[~batch_complete].mean()
                    n_types_cls += 1
                if n_types_cls > 0:
                    cls_loss /= n_types_cls
                    
                if lambda_align > 0 and batch_complete.any():
                    # Since align is only on complete, average by sample type is just its mean
                    align_loss = align_sample[batch_complete].mean()
                else:
                    align_loss = torch.tensor(0.0, device=device)
                    
                if lambda_router > 0 and isinstance(model, MixtureFeatureAdapter) and valid.any():
                    router_sample = torch.nn.functional.cross_entropy(details["router_logits"][valid], batch_cluster[valid], reduction="none")
                    router_loss = router_sample.mean()
                else:
                    router_sample = torch.zeros(int(valid.sum().item()), device=device)
                    router_loss = torch.tensor(0.0, device=device)
            else:
                cls_loss = cls_sample.mean()
                if lambda_align > 0 and batch_complete.any():
                    align_loss = align_sample[batch_complete].mean()
                else:
                    align_loss = torch.tensor(0.0, device=device)
                if lambda_router > 0 and isinstance(model, MixtureFeatureAdapter) and valid.any():
                    router_sample = torch.nn.functional.cross_entropy(details["router_logits"][valid], batch_cluster[valid], reduction="none")
                    router_loss = router_sample.mean()
                else:
                    router_sample = torch.zeros(int(valid.sum().item()), device=device)
                    router_loss = torch.tensor(0.0, device=device)
            loss = cls_loss + lambda_align * align_loss + lambda_router * router_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            complete_cls = cls_sample[batch_complete]
            incomplete_cls = cls_sample[~batch_complete]
            complete_align = align_sample[batch_complete] if batch_complete.any() else torch.zeros(0, device=device)
            complete_router = router_sample if valid.any() else torch.zeros(0, device=device)

            weight_stats["complete"]["n"] += int(batch_complete.sum().item())
            weight_stats["incomplete"]["n"] += int((~batch_complete).sum().item())
            if complete_cls.numel():
                weight_stats["complete"]["cls"] += float(complete_cls.sum().detach().cpu())
                weight_stats["complete"]["align"] += float(complete_align.sum().detach().cpu())
                if complete_router.numel():
                    weight_stats["complete"]["router"] += float(complete_router.sum().detach().cpu())
                weight_stats["complete"]["total"] += float(
                    (complete_cls + lambda_align * complete_align + lambda_router * (complete_router.mean() if complete_router.numel() else 0.0)).sum().detach().cpu()
                )
            if incomplete_cls.numel():
                weight_stats["incomplete"]["cls"] += float(incomplete_cls.sum().detach().cpu())
                weight_stats["incomplete"]["total"] += float(incomplete_cls.sum().detach().cpu())

        val_logits = val_logits_builder(model, routing_mode="learned")
        threshold = select_best_thresholds(val_logits, y_val, args.threshold_grid, args.threshold_strategies, args.tune_thresholds)
        training_rows.append(
            {
                "protocol": protocol,
                "seed": seed,
                "sampling_mode": args.complete_sampling,
                "stage": "adapter",
                "method": method,
                "branch": branch,
                "epoch": epoch,
                "lambda_align": lambda_align,
                "lambda_router": lambda_router,
                "train_loss": float(np.mean(losses)),
                "val_macro_f1": threshold["metrics"]["macro_f1"],
                "val_micro_f1": threshold["metrics"]["micro_f1"],
                "val_sample_f1": threshold["metrics"]["sample_f1"],
                "val_bce_loss": threshold["metrics"]["bce_loss"],
                "threshold_strategy": threshold["threshold_mode"],
            }
        )
        if threshold["metrics"]["macro_f1"] > best_score:
            best_score = float(threshold["metrics"]["macro_f1"])
            best_threshold = threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    weight_rows = []
    incomplete_weight = None
    if weight_stats["incomplete"]["n"] > 0:
        incomplete_weight = weight_stats["incomplete"]["total"] / weight_stats["incomplete"]["n"]
    for sample_type in ("complete", "incomplete"):
        count = max(weight_stats[sample_type]["n"], 1)
        effective = weight_stats[sample_type]["total"] / count
        relative = effective / incomplete_weight if incomplete_weight not in {None, 0.0} else float("nan")
        weight_rows.append(
            {
                "seed": seed,
                "protocol": protocol,
                "sampling_mode": args.complete_sampling,
                "method": method,
                "branch": branch,
                "lambda_align": lambda_align,
                "lambda_router": lambda_router,
                "sample_type": sample_type,
                "n_samples": int(weight_stats[sample_type]["n"]),
                "avg_cls_loss": weight_stats[sample_type]["cls"] / count,
                "avg_align_loss": weight_stats[sample_type]["align"] / count,
                "avg_router_loss": weight_stats[sample_type]["router"] / count,
                "effective_loss_weight": effective,
                "relative_to_incomplete": relative,
            }
        )
    return model, best_threshold, weight_rows


def combine_logits(
    split: str,
    table: np.ndarray,
    full_logits: np.ndarray,
    branch_logits: Dict[str, np.ndarray],
    y: np.ndarray,
    complete_eval_mode: str = "branch_ensemble",
) -> np.ndarray:
    logits = np.zeros((len(table), y.shape[1]), dtype=np.float32)
    complete = table == MISSING_COMPLETE
    image_missing = table == MISSING_IMAGE
    text_missing = table == MISSING_TEXT

    def branch_ensemble(mask: np.ndarray) -> np.ndarray:
        sources = [value for value in branch_logits.values() if value is not None]
        if not sources:
            # Emergency fallback only. Normal deployable evaluation should always
            # provide at least one missing-type branch, keeping the teacher out of
            # the final classifier path.
            return full_logits[mask]
        return np.stack([source[mask] for source in sources], axis=0).mean(axis=0)

    if complete.any():
        if complete_eval_mode == "full_teacher":
            logits[complete] = full_logits[complete]
        else:
            logits[complete] = branch_ensemble(complete)

    if image_missing.any():
        source = branch_logits.get("image_missing")
        logits[image_missing] = source[image_missing] if source is not None else branch_ensemble(image_missing)
    if text_missing.any():
        source = branch_logits.get("text_missing")
        logits[text_missing] = source[text_missing] if source is not None else branch_ensemble(text_missing)
    return logits


def eval_rows_for_subgroups(
    args: argparse.Namespace,
    protocol: str,
    seed: int,
    method: str,
    routing_mode: str,
    logits: np.ndarray,
    y: np.ndarray,
    table: np.ndarray,
    threshold: Dict,
    deployable: bool,
) -> List[Dict]:
    groups = {
        "overall": np.ones(len(table), dtype=bool),
        "complete": table == MISSING_COMPLETE,
        "image_missing": table == MISSING_IMAGE,
        "text_missing": table == MISSING_TEXT,
    }
    rows = []
    for subgroup, mask in groups.items():
        if not mask.any():
            continue
        metrics = metrics_for_logits(logits[mask], y[mask], threshold, table[mask])
        rows.append(
            {
                "dataset": args.dataset,
                "protocol": protocol,
                "seed": seed,
                "sampling_mode": args.complete_sampling,
                "method": method,
                "routing_mode": routing_mode,
                "split": "test",
                "subgroup": subgroup,
                "n_samples": int(mask.sum()),
                "deployable": deployable,
                **metrics,
            }
        )
    return rows


def alignment_row(
    protocol: str,
    seed: int,
    method: str,
    routing_mode: str,
    split: str,
    branch: str,
    z_full: np.ndarray,
    z_before: np.ndarray,
    z_after: np.ndarray,
    mask: np.ndarray,
) -> Optional[Dict]:
    if not mask.any():
        return None
    before = z_before[mask]
    after = z_after[mask]
    target = z_full[mask]
    mse_before = float(np.mean((before - target) ** 2))
    mse_after = float(np.mean((after - target) ** 2))
    cos_before = cosine_mean(before, target)
    cos_after = cosine_mean(after, target)
    return {
        "split": split,
        "protocol": protocol,
        "seed": seed,
        "method": method,
        "routing_mode": routing_mode,
        "branch": branch,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "cosine_before": cos_before,
        "cosine_after": cos_after,
        "delta_mse": mse_before - mse_after,
        "delta_cosine": cos_after - cos_before,
    }


def router_stats_row(
    protocol: str,
    seed: int,
    method: str,
    branch: str,
    split: str,
    cluster_true: np.ndarray,
    alpha: np.ndarray,
    phase: str,
    lambda_router: float,
    scope: str = "all",
    mask: Optional[np.ndarray] = None,
) -> Dict:
    if mask is not None:
        cluster_true = cluster_true[mask]
        alpha = alpha[mask]
    pred = alpha.argmax(axis=1)
    entropy = -(alpha * np.log(np.clip(alpha, 1e-8, 1.0))).sum(axis=1)
    return {
        "dataset": "mmimdb",
        "protocol": protocol,
        "seed": seed,
        "method": method,
        "branch": branch,
        "split": split,
        "phase": phase,
        "scope": scope,
        "lambda_router": lambda_router,
        "router_accuracy": float((pred == cluster_true).mean()),
        "router_macro_f1": float(f1_score(cluster_true, pred, average="macro", zero_division=0)),
        "router_entropy_mean": float(entropy.mean()),
        "alpha_max_mean": float(alpha.max(axis=1).mean()),
    }


def serialize_thresholds(thresholds) -> str:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value
    return json.dumps(convert(thresholds))


def threshold_array_from_info(threshold_info: Dict, subgroup: str = "overall") -> np.ndarray:
    thresholds = threshold_info["thresholds"]
    if isinstance(thresholds, dict):
        return np.asarray(thresholds.get(subgroup, thresholds.get("overall")), dtype=np.float32)
    return np.asarray(thresholds, dtype=np.float32)


def apply_thresholds_protocol(logits: np.ndarray, table: Optional[np.ndarray], threshold_info: Dict) -> np.ndarray:
    thresholds = threshold_info["thresholds"]
    if not isinstance(thresholds, dict) or table is None:
        return apply_thresholds(sigmoid(logits), threshold_array_from_info(threshold_info, "overall"))
    pred = np.zeros_like(logits, dtype=np.int64)
    for subgroup, code in (("complete", MISSING_COMPLETE), ("image_missing", MISSING_IMAGE), ("text_missing", MISSING_TEXT)):
        mask = table == code
        if mask.any():
            pred[mask] = apply_thresholds(sigmoid(logits[mask]), threshold_array_from_info(threshold_info, subgroup))
    return pred


def metrics_for_logits(logits: np.ndarray, y: np.ndarray, threshold_info: Dict, table: Optional[np.ndarray] = None) -> Dict:
    if isinstance(threshold_info.get("thresholds"), dict):
        pred = apply_thresholds_protocol(logits, table, threshold_info)
        return {
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
            "sample_f1": float(f1_score(y, pred, average="samples", zero_division=0)),
            "accuracy": float((pred == y).all(axis=1).mean()),
            "bce_loss": mean_bce_loss(logits, y),
        }
    return multilabel_metrics(logits, y, threshold_info["thresholds"])


def select_protocol_thresholds(
    val_logits: np.ndarray,
    val_y: np.ndarray,
    val_table: np.ndarray,
    args: argparse.Namespace,
    protocol: str,
    seed: int,
    method: str,
    routing_mode: str,
    accum: Optional[Dict[str, List[Dict]]] = None,
    min_group_size: int = 20,
) -> Dict:
    overall = select_best_thresholds(val_logits, val_y, args.threshold_grid, args.threshold_strategies, args.tune_thresholds)
    if not getattr(args, "per_branch_thresholds", False):
        return overall
    thresholds = {"overall": overall["thresholds"]}
    rows = []
    for subgroup, code in (("complete", MISSING_COMPLETE), ("image_missing", MISSING_IMAGE), ("text_missing", MISSING_TEXT)):
        mask = val_table == code
        fallback = bool(mask.sum() < min_group_size)
        if fallback:
            info = overall
        else:
            info = select_best_thresholds(val_logits[mask], val_y[mask], args.threshold_grid, args.threshold_strategies, args.tune_thresholds)
        thresholds[subgroup] = info["thresholds"]
        rows.append({
            "protocol": protocol,
            "seed": seed,
            "sampling_mode": args.complete_sampling,
            "method": method,
            "routing_mode": routing_mode,
            "subgroup": subgroup,
            "threshold_strategy": info["threshold_mode"],
            "threshold": serialize_thresholds(info["thresholds"]),
            "val_macro_f1": info["metrics"]["macro_f1"],
            "val_micro_f1": info["metrics"]["micro_f1"],
            "val_sample_f1": info["metrics"]["sample_f1"],
            "val_bce_loss": info["metrics"]["bce_loss"],
            "n_val_samples": int(mask.sum()),
            "fallback_to_overall": fallback,
        })
    threshold_info = {
        "threshold_mode": "per_branch",
        "thresholds": thresholds,
        "metrics": metrics_for_logits(val_logits, val_y, {"threshold_mode": "per_branch", "thresholds": thresholds}, val_table),
    }
    if accum is not None:
        accum.setdefault("branch_thresholds", []).extend(rows)
    return threshold_info


def threshold_row(
    protocol: str,
    seed: int,
    sampling_mode: str,
    method: str,
    routing_mode: str,
    threshold_info: Dict,
    deployable: bool,
    uses_full_modality: bool,
    uses_test_label: bool,
    temperature: Optional[float],
) -> Dict:
    return {
        "protocol": protocol,
        "seed": seed,
        "sampling_mode": sampling_mode,
        "method": method,
        "routing_mode": routing_mode,
        "temperature": "" if temperature is None else temperature,
        "deployable": deployable,
        "uses_full_modality": uses_full_modality,
        "uses_test_label": uses_test_label,
        "threshold_strategy": threshold_info["threshold_mode"],
        "threshold": serialize_thresholds(threshold_info["thresholds"]),
        "val_macro_f1": threshold_info["metrics"]["macro_f1"],
        "val_micro_f1": threshold_info["metrics"]["micro_f1"],
        "val_sample_f1": threshold_info["metrics"]["sample_f1"],
        "val_bce_loss": threshold_info["metrics"]["bce_loss"],
    }


def result_row(
    args: argparse.Namespace,
    protocol: str,
    seed: int,
    method: str,
    routing_mode: str,
    threshold_info: Dict,
    logits: np.ndarray,
    y: np.ndarray,
    summary: Dict,
    sampling_mode: str,
    lambda_align: float,
    lambda_router: float,
    deployable: bool,
    uses_full_modality: bool,
    uses_test_label: bool,
    temperature: Optional[float],
    table: Optional[np.ndarray] = None,
) -> Dict:
    metrics = metrics_for_logits(logits, y, threshold_info, table)
    return {
        "dataset": args.dataset,
        "protocol": protocol,
        "seed": seed,
        "sampling_mode": sampling_mode,
        "method": method,
        "routing_mode": routing_mode,
        "temperature": "" if temperature is None else temperature,
        "lambda_align": lambda_align,
        "lambda_router": lambda_router,
        "missing_ratio": args.missing_ratio,
        "complete_ratio": summary["complete_ratio"],
        "image_missing_ratio": summary["image_missing_ratio"],
        "text_missing_ratio": summary["text_missing_ratio"],
        "deployable": deployable,
        "uses_full_modality": uses_full_modality,
        "uses_test_label": uses_test_label,
        "threshold_strategy": threshold_info["threshold_mode"],
        "threshold": serialize_thresholds(threshold_info["thresholds"]),
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "sample_f1": metrics["sample_f1"],
        "accuracy": metrics["accuracy"],
        "bce_loss": metrics["bce_loss"],
    }


def positive_label_ids(y_row: np.ndarray) -> List[int]:
    return np.flatnonzero(y_row > 0.5).astype(int).tolist()


def per_sample_diag_rows(
    split: str,
    protocol: str,
    seed: int,
    method: str,
    routing_mode: str,
    table: np.ndarray,
    sample_ids: np.ndarray,
    y: np.ndarray,
    payload: Dict[str, np.ndarray],
    threshold_info: Dict,
    missing_threshold: Dict,
    cluster_labels: Optional[np.ndarray],
    residual_norm: Optional[np.ndarray],
    z_full: Optional[np.ndarray],
) -> List[Dict]:
    pred_after = apply_thresholds_protocol(payload["logits"], table, threshold_info)
    pred_before = apply_thresholds_protocol(payload["base_logits"], table, missing_threshold)
    alpha = payload["alpha"]
    alpha_entropy = -(alpha * np.log(np.clip(alpha, 1e-8, 1.0))).sum(axis=1)
    delta_norm = np.linalg.norm(payload["delta"], axis=1)
    static_delta_norm = np.linalg.norm(payload.get("static_delta", np.zeros_like(payload["delta"])), axis=1)
    mixture_delta_norm = np.linalg.norm(payload.get("mixture_delta", payload["delta"]), axis=1)
    beta_values = payload.get("beta", np.ones((len(payload["logits"]), 1), dtype=np.float32))
    beta_mean = beta_values.mean(axis=1) if beta_values.ndim > 1 else beta_values
    probs = sigmoid(payload["logits"])
    eps = 1e-8
    bce = -(
        y * np.log(np.clip(probs, eps, 1.0 - eps))
        + (1.0 - y) * np.log(np.clip(1.0 - probs, eps, 1.0))
    ).mean(axis=1)
    rows = []
    for idx in range(len(sample_ids)):
        row = {
            "sample_id": str(sample_ids[idx]),
            "split": split,
            "seed": seed,
            "protocol": protocol,
            "method": method,
            "routing_mode": routing_mode,
            "subgroup": subgroup_name(int(table[idx])),
            "is_complete": bool(table[idx] == MISSING_COMPLETE),
            "true_labels": json.dumps(positive_label_ids(y[idx])),
            "predicted_labels": json.dumps(positive_label_ids(pred_after[idx])),
            "bce_loss": float(bce[idx]),
            "macro_relevant_error_count": int(np.abs(pred_after[idx] - y[idx]).sum()),
            "alpha_entropy": float(alpha_entropy[idx]),
            "alpha_max": float(alpha[idx].max()),
            "selected_expert": int(alpha[idx].argmax()),
            "cluster_label_if_available": int(cluster_labels[idx]) if cluster_labels is not None else "",
            "residual_norm_if_available": float(residual_norm[idx]) if residual_norm is not None else float("nan"),
            "delta_norm": float(delta_norm[idx]),
            "static_delta_norm": float(static_delta_norm[idx]),
            "mixture_delta_norm": float(mixture_delta_norm[idx]),
            "final_delta_norm": float(delta_norm[idx]),
            "beta": float(beta_mean[idx]),
            "beta_mean_if_vector": float(beta_mean[idx]),
            "correct_before_adapter": bool(np.array_equal(pred_before[idx], y[idx])),
            "correct_after_adapter": bool(np.array_equal(pred_after[idx], y[idx])),
        }
        if z_full is not None:
            row["z_static_alignment_mse_if_available"] = float(np.mean((payload.get("z_static", payload["z_before"])[idx] - z_full[idx]) ** 2))
            row["z_static_alignment_cosine_if_available"] = float(
                np.dot(payload.get("z_static", payload["z_before"])[idx], z_full[idx])
                / (np.linalg.norm(payload.get("z_static", payload["z_before"])[idx]) * np.linalg.norm(z_full[idx]) + 1e-12)
            )
            row["z_final_alignment_mse_if_available"] = float(np.mean((payload["z_after"][idx] - z_full[idx]) ** 2))
            row["z_final_alignment_cosine_if_available"] = float(
                np.dot(payload["z_after"][idx], z_full[idx])
                / (np.linalg.norm(payload["z_after"][idx]) * np.linalg.norm(z_full[idx]) + 1e-12)
            )
        else:
            row["z_static_alignment_mse_if_available"] = float("nan")
            row["z_static_alignment_cosine_if_available"] = float("nan")
            row["z_final_alignment_mse_if_available"] = float("nan")
            row["z_final_alignment_cosine_if_available"] = float("nan")
            row["z_alignment_mse_if_available"] = float("nan")
            row["z_alignment_cosine_if_available"] = float("nan")
        for expert_id in range(alpha.shape[1]):
            row[f"alpha_{expert_id}"] = float(alpha[idx, expert_id])
        rows.append(row)
    return rows


def beta_summary_rows(
    args: argparse.Namespace,
    protocol: str,
    seed: int,
    method: str,
    split: str,
    branch: str,
    table: np.ndarray,
    payload: Dict[str, np.ndarray],
    logits: np.ndarray,
    y: np.ndarray,
    threshold_info: Dict,
) -> List[Dict]:
    beta = payload.get("beta", np.ones((len(logits), 1), dtype=np.float32))
    beta_1d = beta.mean(axis=1) if beta.ndim > 1 else beta
    alpha = payload.get("alpha", np.ones((len(logits), 1), dtype=np.float32))
    alpha_entropy = -(alpha * np.log(np.clip(alpha, 1e-8, 1.0))).sum(axis=1)
    static_delta_norm = np.linalg.norm(payload.get("static_delta", np.zeros_like(payload["delta"])), axis=1)
    mixture_delta_norm = np.linalg.norm(payload.get("mixture_delta", payload["delta"]), axis=1)
    final_delta_norm = np.linalg.norm(payload["delta"], axis=1)
    rows = []
    for subgroup, mask in {
        "overall": np.ones(len(table), dtype=bool),
        "complete": table == MISSING_COMPLETE,
        "image_missing": table == MISSING_IMAGE,
        "text_missing": table == MISSING_TEXT,
    }.items():
        if not mask.any():
            continue
        metrics = metrics_for_logits(logits[mask], y[mask], threshold_info, table[mask])
        rows.append({
            "protocol": protocol,
            "seed": seed,
            "method": method,
            "split": split,
            "subgroup": subgroup,
            "branch": branch,
            "beta_mean": float(beta_1d[mask].mean()),
            "beta_std": float(beta_1d[mask].std()),
            "beta_min": float(beta_1d[mask].min()),
            "beta_max": float(beta_1d[mask].max()),
            "beta_q25": float(np.quantile(beta_1d[mask], 0.25)),
            "beta_q50": float(np.quantile(beta_1d[mask], 0.50)),
            "beta_q75": float(np.quantile(beta_1d[mask], 0.75)),
            "alpha_entropy_mean": float(alpha_entropy[mask].mean()),
            "alpha_max_mean": float(alpha[mask].max(axis=1).mean()),
            "static_delta_norm_mean": float(static_delta_norm[mask].mean()),
            "mixture_delta_norm_mean": float(mixture_delta_norm[mask].mean()),
            "final_delta_norm_mean": float(final_delta_norm[mask].mean()),
            "macro_f1": metrics["macro_f1"],
            "bce_loss": metrics["bce_loss"],
        })
    return rows


def save_npz_diagnostics(
    output_dir: Path,
    protocol: str,
    seed: int,
    method: str,
    routing_mode: str,
    split: str,
    payload: Dict[str, np.ndarray],
) -> None:
    diag_dir = ensure_dir(output_dir / "diagnostic_npz")
    path = diag_dir / f"{protocol}_seed{seed}_{method}_{routing_mode}_{split}.npz"
    np.savez_compressed(
        path,
        logits=payload["logits"],
        z_before=payload["z_before"],
        z_static=payload.get("z_static", payload["z_before"]),
        z_after=payload["z_after"],
        delta=payload["delta"],
        static_delta=payload.get("static_delta", np.zeros_like(payload["delta"])),
        mixture_delta=payload.get("mixture_delta", payload["delta"]),
        beta=payload.get("beta", np.ones((len(payload["logits"]), 1), dtype=np.float32)),
        base_logits=payload["base_logits"],
        alpha=payload["alpha"],
        router_logits=payload.get("router_logits", np.zeros((len(payload["logits"]), 0), dtype=np.float32)),
    )


def load_reference_macro(reference_df: pd.DataFrame, protocol: str, seed: int, method: str) -> float:
    if reference_df.empty:
        return float("nan")
    sub = reference_df[(reference_df["protocol"] == protocol) & (reference_df["seed"] == seed) & (reference_df["method"] == method)]
    if sub.empty:
        return float("nan")
    return float(sub.sort_values("macro_f1", ascending=False).iloc[0]["macro_f1"])


def load_reference_router_stats(reference_router: pd.DataFrame, protocol: str, seed: int, method: str) -> Tuple[float, float]:
    if reference_router.empty:
        return float("nan"), float("nan")
    sub = reference_router[(reference_router["protocol"] == protocol) & (reference_router["seed"] == seed) & (reference_router["method"] == method)]
    if sub.empty:
        return float("nan"), float("nan")
    row = sub.iloc[0]
    return float(row["router_entropy_mean"]), float(row["router_accuracy"])


def save_checkpoint(
    output_dir: Path,
    protocol: str,
    seed: int,
    method: str,
    branch_models: Dict[str, nn.Module],
    branch_centers: Dict[str, np.ndarray],
    threshold_info: Dict,
    lambda_router: float,
    lambda_align: float,
) -> None:
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    payload = {
        "protocol": protocol,
        "seed": seed,
        "method": method,
        "lambda_router": lambda_router,
        "lambda_align": lambda_align,
        "threshold_mode": threshold_info["threshold_mode"],
        "thresholds": threshold_info["thresholds"],
        "branch_centers": branch_centers,
        "branch_states": {branch: model.state_dict() for branch, model in branch_models.items()},
    }
    torch.save(payload, checkpoint_dir / f"{protocol}_seed{seed}_{method}_lambda{lambda_router:.4f}.pt")


def generate_matched_oracle_rows(
    args: argparse.Namespace,
    protocol: str,
    seed: int,
    branch: str,
    branch_model: MixtureFeatureAdapter,
    branch_centers: np.ndarray,
    branch_clusters: Dict[str, np.ndarray],
    branch_inputs: Dict[str, np.ndarray],
    z_full: Dict[str, np.ndarray],
    tables: Dict[str, np.ndarray],
    full_logits: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    summary: Dict,
    missing_threshold: Dict,
    lambda_align: float,
    lambda_router: float,
    accum: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    learned_train = predict_adapter_mode(branch_model, branch_inputs["train"], accum["device"])
    learned_val = predict_adapter_mode(branch_model, branch_inputs["val"], accum["device"])
    learned_test = predict_adapter_mode(branch_model, branch_inputs["test"], accum["device"])
    learned_val_combined = combine_logits(
        "val",
        tables["val"],
        full_logits["val"],
        {branch: learned_val["logits"]},
        labels["val"],
        args.complete_eval_mode,
    )
    learned_test_combined = combine_logits(
        "test",
        tables["test"],
        full_logits["test"],
        {branch: learned_test["logits"]},
        labels["test"],
        args.complete_eval_mode,
    )
    learned_threshold = select_best_thresholds(learned_val_combined, labels["val"], args.threshold_grid, args.threshold_strategies, args.tune_thresholds)
    learned_row = result_row(
        args,
        protocol,
        seed,
        "prototype_plain_hard",
        "learned_router",
        learned_threshold,
        learned_test_combined,
        labels["test"],
        summary,
        args.complete_sampling,
        lambda_align,
        lambda_router,
        True,
        False,
        False,
        None,
        tables["test"],
    )
    learned_row["checkpoint_method"] = "prototype_plain_hard"
    accum["matched_oracle"].append(learned_row)
    accum["thresholds"].append(
        threshold_row(protocol, seed, args.complete_sampling, "prototype_plain_hard", "learned_router", learned_threshold, True, False, False, None)
    )
    if args.save_prototype_diagnostics:
        residual_train = np.linalg.norm(z_full["train"] - learned_train["z_before"], axis=1)
        cluster_val = branch_clusters["val"]
        cluster_test = branch_clusters["test"]
        residual_val = np.linalg.norm(z_full["val"] - learned_val["z_before"], axis=1)
        residual_test = np.linalg.norm(z_full["test"] - learned_test["z_before"], axis=1)
        accum["per_sample"].extend(
            per_sample_diag_rows(
                "train",
                protocol,
                seed,
                "prototype_plain_hard",
                "learned_router",
                tables["train"],
                accum["sample_ids"]["train"],
                labels["train"],
                learned_train,
                learned_threshold,
                missing_threshold,
                branch_clusters["train"],
                residual_train,
                z_full["train"],
            )
        )
        accum["per_sample"].extend(
            per_sample_diag_rows(
                "val",
                protocol,
                seed,
                "prototype_plain_hard",
                "learned_router",
                tables["val"],
                accum["sample_ids"]["val"],
                labels["val"],
                learned_val,
                learned_threshold,
                missing_threshold,
                cluster_val,
                residual_val,
                z_full["val"],
            )
        )
        accum["per_sample"].extend(
            per_sample_diag_rows(
                "test",
                protocol,
                seed,
                "prototype_plain_hard",
                "learned_router",
                tables["test"],
                accum["sample_ids"]["test"],
                labels["test"],
                learned_test,
                learned_threshold,
                missing_threshold,
                cluster_test,
                residual_test,
                z_full["test"],
            )
        )
        save_npz_diagnostics(output_dir, protocol, seed, "prototype_plain_hard", "learned_router", "train", learned_train)
        save_npz_diagnostics(output_dir, protocol, seed, "prototype_plain_hard", "learned_router", "val", learned_val)
        save_npz_diagnostics(output_dir, protocol, seed, "prototype_plain_hard", "learned_router", "test", learned_test)

    learned_macro = learned_row["macro_f1"]

    hard_val = predict_adapter_mode(branch_model, branch_inputs["val"], accum["device"], cluster_override=branch_clusters["val"])
    hard_test = predict_adapter_mode(branch_model, branch_inputs["test"], accum["device"], cluster_override=branch_clusters["test"])
    hard_val_combined = combine_logits(
        "val",
        tables["val"],
        full_logits["val"],
        {branch: hard_val["logits"]},
        labels["val"],
        args.complete_eval_mode,
    )
    hard_test_combined = combine_logits(
        "test",
        tables["test"],
        full_logits["test"],
        {branch: hard_test["logits"]},
        labels["test"],
        args.complete_eval_mode,
    )
    hard_threshold = select_best_thresholds(hard_val_combined, labels["val"], args.threshold_grid, args.threshold_strategies, args.tune_thresholds)
    hard_row = result_row(
        args,
        protocol,
        seed,
        "prototype_plain_hard",
        "oracle_hard_same_checkpoint",
        hard_threshold,
        hard_test_combined,
        labels["test"],
        summary,
        args.complete_sampling,
        lambda_align,
        lambda_router,
        False,
        True,
        False,
        None,
        tables["test"],
    )
    hard_row["checkpoint_method"] = "prototype_plain_hard"
    hard_row["delta_vs_learned_router"] = hard_row["macro_f1"] - learned_macro
    hard_row["interpretation"] = "Matched hard oracle on same experts/classifier."
    accum["matched_oracle"].append(hard_row)
    accum["thresholds"].append(
        threshold_row(protocol, seed, args.complete_sampling, "prototype_plain_hard", "oracle_hard_same_checkpoint", hard_threshold, False, True, False, None)
    )
    if args.save_prototype_diagnostics:
        residual_val = np.linalg.norm(z_full["val"] - hard_val["z_before"], axis=1)
        residual_test = np.linalg.norm(z_full["test"] - hard_test["z_before"], axis=1)
        accum["per_sample"].extend(
            per_sample_diag_rows(
                "test",
                protocol,
                seed,
                "prototype_plain_hard",
                "oracle_hard_same_checkpoint",
                tables["test"],
                accum["sample_ids"]["test"],
                labels["test"],
                hard_test,
                hard_threshold,
                missing_threshold,
                branch_clusters["test"],
                residual_test,
                z_full["test"],
            )
        )
        save_npz_diagnostics(output_dir, protocol, seed, "prototype_plain_hard", "oracle_hard_same_checkpoint", "test", hard_test)

    for temperature in args.oracle_soft_temperatures:
        alpha_val = soft_cluster_alpha(z_full["val"] - learned_val["z_before"], branch_centers, temperature)
        alpha_test = soft_cluster_alpha(z_full["test"] - learned_test["z_before"], branch_centers, temperature)
        soft_val = predict_adapter_mode(branch_model, branch_inputs["val"], accum["device"], alpha_override=alpha_val)
        soft_test = predict_adapter_mode(branch_model, branch_inputs["test"], accum["device"], alpha_override=alpha_test)
        soft_val_combined = combine_logits(
            "val",
            tables["val"],
            full_logits["val"],
            {branch: soft_val["logits"]},
            labels["val"],
            args.complete_eval_mode,
        )
        soft_combined = combine_logits(
            "test",
            tables["test"],
            full_logits["test"],
            {branch: soft_test["logits"]},
            labels["test"],
            args.complete_eval_mode,
        )
        soft_threshold = select_best_thresholds(soft_val_combined, labels["val"], args.threshold_grid, args.threshold_strategies, args.tune_thresholds)
        soft_row = result_row(
            args,
            protocol,
            seed,
            "prototype_plain_hard",
            "oracle_soft_same_checkpoint",
            soft_threshold,
            soft_combined,
            labels["test"],
            summary,
            args.complete_sampling,
            lambda_align,
            lambda_router,
            False,
            True,
            False,
            temperature,
            tables["test"],
        )
        soft_row["checkpoint_method"] = "prototype_plain_hard"
        soft_row["delta_vs_learned_router"] = soft_row["macro_f1"] - learned_macro
        soft_row["interpretation"] = "Matched soft oracle on same experts/classifier."
        accum["matched_oracle"].append(soft_row)
        accum["thresholds"].append(
            threshold_row(protocol, seed, args.complete_sampling, "prototype_plain_hard", "oracle_soft_same_checkpoint", soft_threshold, False, True, False, temperature)
        )
        if args.save_prototype_diagnostics:
            residual_test = np.linalg.norm(z_full["test"] - soft_test["z_before"], axis=1)
            accum["per_sample"].extend(
                per_sample_diag_rows(
                    "test",
                    protocol,
                    seed,
                    "prototype_plain_hard",
                    f"oracle_soft_same_checkpoint_T{temperature}",
                    tables["test"],
                    accum["sample_ids"]["test"],
                    labels["test"],
                    soft_test,
                    soft_threshold,
                    missing_threshold,
                    branch_clusters["test"],
                    residual_test,
                    z_full["test"],
                )
            )
            save_npz_diagnostics(output_dir, protocol, seed, "prototype_plain_hard", f"oracle_soft_same_checkpoint_T{temperature}", "test", soft_test)


def evaluate_and_record_method(
    args: argparse.Namespace,
    protocol: str,
    seed: int,
    method: str,
    routing_mode: str,
    lambda_align: float,
    lambda_router: float,
    threshold: Dict,
    val_logits: np.ndarray,
    test_logits: np.ndarray,
    labels: Dict[str, np.ndarray],
    tables: Dict[str, np.ndarray],
    full_logits: Dict[str, np.ndarray],
    summary: Dict,
    accum: Dict[str, List[Dict]],
    deployable: bool,
    uses_full_modality: bool,
    uses_test_label: bool,
    temperature: Optional[float] = None,
) -> None:
    row = result_row(
        args,
        protocol,
        seed,
        method,
        routing_mode,
        threshold,
        test_logits,
        labels["test"],
        summary,
        args.complete_sampling,
        lambda_align,
        lambda_router,
        deployable,
        uses_full_modality,
        uses_test_label,
        temperature,
        tables["test"],
    )
    accum["results"].append(row)
    accum["subgroups"].extend(eval_rows_for_subgroups(args, protocol, seed, method, routing_mode, test_logits, labels["test"], tables["test"], threshold, deployable))
    accum["thresholds"].append(threshold_row(protocol, seed, args.complete_sampling, method, routing_mode, threshold, deployable, uses_full_modality, uses_test_label, temperature))


def run_protocol_seed(args, protocol: str, seed: int, shared: Dict, output_dir: Path, reference: Dict[str, pd.DataFrame], accum: Dict[str, List[Dict]]) -> None:
    set_seed(seed)
    args.seed = seed
    device = shared["device"]
    features = shared["features"]
    labels = shared["labels"]
    split_meta = shared["split_meta"]
    class_names = shared["class_names"]
    output_dim = labels["train"].shape[1]
    text_dim = features["train"]["text"].shape[1]
    image_dim = features["train"]["image"].shape[1]
    full_dim = image_dim + text_dim
    primary = shared["primary_labels"]

    tables = {}
    table_dir = ensure_dir(output_dir / "missing_tables")
    warnings_rows = []
    label_rows = []
    for split in ("train", "val", "test"):
        table, warnings = make_missing_table(
            len(labels[split]),
            protocol,
            args.missing_ratio,
            args.both_ratio,
            seed,
            split,
            args.complete_sampling,
            primary.get(split),
            args.min_complete_per_label,
        )
        tables[split] = table
        suffix = f"{split}_{protocol}_{args.complete_sampling}_seed{seed}.npy" if args.complete_sampling != "random" else f"{split}_{protocol}_seed{seed}.npy"
        np.save(table_dir / suffix, table)
        accum["missing_tables"].append(missing_table_summary(table, split, protocol, seed, args.complete_sampling))
        label_rows.extend(label_distribution_rows(protocol, seed, split, args.complete_sampling, table, labels[split], class_names, args.min_complete_per_label))
        for warning in warnings:
            warnings_rows.append({"protocol": protocol, "seed": seed, "split": split, "sampling_mode": args.complete_sampling, "warning": warning})

    complete_train = tables["train"] == MISSING_COMPLETE
    complete_val = tables["val"] == MISSING_COMPLETE
    if complete_train.sum() < 5:
        raise ValueError(f"Too few complete train samples for {protocol} seed={seed}: {int(complete_train.sum())}")
    if complete_val.sum() < 5:
        complete_val = np.ones_like(complete_val, dtype=bool)

    full_by_split = {split: concat_full(features[split]) for split in ("train", "val", "test")}
    full_teacher = RepresentationClassifier(full_dim, args.latent_dim, args.hidden_dim, output_dim, args.dropout)
    full_teacher, teacher_threshold = train_branch_base(
        full_teacher,
        full_by_split["train"][complete_train],
        labels["train"][complete_train],
        full_by_split["val"][complete_val],
        labels["val"][complete_val],
        args,
        device,
        seed,
        f"{protocol}_full_teacher",
        protocol,
        accum["training"],
    )
    full_logits, z_full = {}, {}
    for split in ("train", "val", "test"):
        full_logits[split], z_full[split] = predict_representation(full_teacher, full_by_split[split], device)
    for split in ("val", "test"):
        mask = complete_val if split == "val" else tables[split] == MISSING_COMPLETE
        if mask.any():
            metrics = multilabel_metrics(full_logits[split][mask], labels[split][mask], teacher_threshold["thresholds"])
            accum["full_teacher"].append({"dataset": args.dataset, "protocol": protocol, "seed": seed, "split": split, "n_samples": int(mask.sum()), **metrics})

    branches = available_branches_for_protocol(protocol)
    branch_models: Dict[str, RepresentationClassifier] = {}
    branch_thresholds: Dict[str, Dict] = {}
    branch_logits: Dict[str, Dict[str, np.ndarray]] = {}
    branch_z: Dict[str, Dict[str, np.ndarray]] = {}
    branch_centers: Dict[str, np.ndarray] = {}
    branch_clusters: Dict[str, Dict[str, np.ndarray]] = {}

    for branch in branches:
        train_mask = branch_train_mask(tables["train"], branch)
        val_mask = branch_train_mask(tables["val"], branch)
        input_dim = text_dim if branch == "image_missing" else image_dim
        model = RepresentationClassifier(input_dim, args.latent_dim, args.hidden_dim, output_dim, args.dropout)
        model, branch_threshold = train_branch_base(
            model,
            branch_feature(features["train"], branch)[train_mask],
            labels["train"][train_mask],
            branch_feature(features["val"], branch)[val_mask],
            labels["val"][val_mask],
            args,
            device,
            seed,
            f"{protocol}_{branch}_base",
            protocol,
            accum["training"],
        )
        branch_models[branch] = model
        branch_thresholds[branch] = branch_threshold
        branch_logits[branch] = {}
        branch_z[branch] = {}
        for split in ("train", "val", "test"):
            payload = predict_branch(model, branch_feature(features[split], branch), device)
            branch_logits[branch][split] = payload["logits"]
            branch_z[branch][split] = payload["z"]
        for split in ("val", "test"):
            eval_mask = branch_train_mask(tables[split], branch)
            if eval_mask.any():
                metrics = multilabel_metrics(branch_logits[branch][split][eval_mask], labels[split][eval_mask], branch_threshold["thresholds"])
                accum["missing_base"].append(
                    {
                        "dataset": args.dataset,
                        "protocol": protocol,
                        "seed": seed,
                        "branch": branch,
                        "split": split,
                        "n_samples": int(eval_mask.sum()),
                        **metrics,
                    }
                )
        residual_train = z_full["train"][complete_train] - branch_z[branch]["train"][complete_train]
        centers = fit_cluster(residual_train, args.prototype_k, seed)
        branch_centers[branch] = centers
        branch_clusters[branch] = {split: assign_cluster(z_full[split] - branch_z[branch][split], centers) for split in ("train", "val", "test")}
        for cluster_id in range(args.prototype_k):
            row = {"dataset": args.dataset, "protocol": protocol, "seed": seed, "branch": branch, "cluster_id": cluster_id}
            for split in ("train", "val", "test"):
                mask = branch_clusters[branch][split] == cluster_id
                row[f"n_{split}"] = int(mask.sum())
                if mask.any():
                    norms = np.linalg.norm(z_full[split][mask] - branch_z[branch][split][mask], axis=1)
                    row[f"{split}_mean_residual_norm"] = float(norms.mean())
            accum["clusters"].append(row)

    def combined_for_method(method_branch_logits: Dict[str, np.ndarray], split: str) -> np.ndarray:
        filled = {}
        for branch in ("image_missing", "text_missing"):
            if branch in method_branch_logits:
                filled[branch] = method_branch_logits[branch]
            elif branch in branch_logits:
                filled[branch] = branch_logits[branch][split]
        return combine_logits(
            split,
            tables[split],
            full_logits[split],
            filled,
            labels[split],
            args.complete_eval_mode,
        )

    missing_val = combined_for_method({branch: branch_logits[branch]["val"] for branch in branches}, "val")
    missing_test = combined_for_method({branch: branch_logits[branch]["test"] for branch in branches}, "test")
    missing_threshold = select_protocol_thresholds(
        missing_val, labels["val"], tables["val"], args, protocol, seed, "missing_only", "learned_router", accum
    )
    summary = missing_table_summary(tables["test"], "test", protocol, seed, args.complete_sampling)
    evaluate_and_record_method(
        args,
        protocol,
        seed,
        "missing_only",
        "learned_router",
        0.0,
        0.0,
        missing_threshold,
        missing_val,
        missing_test,
        labels,
        tables,
        full_logits,
        summary,
        accum,
        True,
        False,
        False,
    )

    all_method_test_macros: Dict[Tuple[str, str, float], float] = {
        ("missing_only", "learned_router", 0.0): metrics_for_logits(
            missing_test,
            labels["test"],
            missing_threshold,
            tables["test"],
        )["macro_f1"]
    }
    static_macro = None
    wide_macro = None

    prototype_router_list = args.lambda_router_list if args.lambda_router_list else [args.lambda_router]
    if "prototype_plain_hard" not in args.methods:
        prototype_router_list = []

    safe_methods = {"safe_mixture_feature_adapter", "safe_mixture_beta_zero", "safe_mixture_beta_one", "safe_mixture_beta_learned", "safe_mixture_beta_biased"}

    for method in args.methods:
        if method == "missing_only":
            continue
        lambda_candidates = prototype_router_list if method == "prototype_plain_hard" else [0.0]
        if method == "mixture_feature_adapter_unsupervised":
            lambda_candidates = [0.0]
        if method in {"static_feature_adapter", "wide_static_feature_adapter"}:
            lambda_candidates = [0.0]
        for lambda_router in lambda_candidates:
            branch_model_outputs = {}
            branch_model_outputs_test = {}
            branch_detail_test = {}
            branch_detail_val = {}
            trained_branch_models = {}
            sample_weight_rows = []
            for branch in branches:
                base = branch_models[branch]
                rank = args.adapter_rank
                cluster_labels = None
                if method == "static_feature_adapter":
                    model = StaticFeatureAdapter(base, args.latent_dim, rank, True)
                    effective_lambda_router = 0.0
                elif method == "wide_static_feature_adapter":
                    model = StaticFeatureAdapter(base, args.latent_dim, args.wide_static_rank, True)
                    effective_lambda_router = 0.0
                elif method == "mixture_feature_adapter_unsupervised":
                    model = MixtureFeatureAdapter(
                        base, args.latent_dim, rank, output_dim, args.mixture_k,
                        True, args.hidden_dim, args.dropout,
                        enable_safe_beta=False,
                        beta_bias=0.0,
                        branch_router_layernorm=args.branch_router_layernorm,
                        branch_str=branch,
                    )
                    effective_lambda_router = 0.0
                elif method in safe_methods:
                    if method in {"safe_mixture_feature_adapter", "safe_mixture_beta_biased"}:
                        beta_bias = args.beta_bias_image if branch == "image_missing" else args.beta_bias_text
                        beta_override_mode = None
                    elif method == "safe_mixture_beta_learned":
                        beta_bias = 0.0
                        beta_override_mode = None
                    elif method == "safe_mixture_beta_zero":
                        beta_bias = 0.0
                        beta_override_mode = "zero"
                    elif method == "safe_mixture_beta_one":
                        beta_bias = 0.0
                        beta_override_mode = "one"
                    model = MixtureFeatureAdapter(
                        base, args.latent_dim, rank, output_dim, args.mixture_k,
                        True, args.hidden_dim, args.dropout,
                        enable_safe_beta=True,
                        beta_bias=beta_bias,
                        branch_router_layernorm=args.branch_router_layernorm,
                        branch_str=branch,
                    )
                    model.beta_override_mode = beta_override_mode
                    effective_lambda_router = 0.0
                elif method == "prototype_plain_hard":
                    model = MixtureFeatureAdapter(
                        base, args.latent_dim, rank, output_dim, args.prototype_k,
                        True, args.hidden_dim, args.dropout,
                        enable_safe_beta=False,
                        beta_bias=0.0,
                        branch_router_layernorm=args.branch_router_layernorm,
                        branch_str=branch,
                    )
                    cluster_labels = np.full(len(labels["train"]), -1, dtype=np.int64)
                    cluster_labels[complete_train] = branch_clusters[branch]["train"][complete_train]
                    effective_lambda_router = lambda_router
                else:
                    continue

                train_mask = branch_train_mask(tables["train"], branch)
                x_train = branch_feature(features["train"], branch)[train_mask]
                y_train = labels["train"][train_mask]
                z_full_train = z_full["train"][train_mask]
                complete_mask_train = complete_train[train_mask]
                cluster_train = cluster_labels[train_mask] if cluster_labels is not None else None

                def val_builder(current_model, routing_mode: str):
                    if routing_mode == "learned":
                        detail = predict_adapter_mode(current_model, branch_feature(features["val"], branch), device)
                    else:
                        raise ValueError(routing_mode)
                    temp = {b: branch_logits[b]["val"] for b in branches}
                    temp[branch] = detail["logits"]
                    return combined_for_method(temp, "val")

                try:
                    model, threshold_info, weight_rows = train_adapter(
                        model,
                        x_train,
                        y_train,
                        z_full_train,
                        complete_mask_train,
                        cluster_train,
                        branch_feature(features["val"], branch),
                        labels["val"],
                        val_builder,
                        args,
                        device,
                        seed,
                        protocol,
                        method,
                        branch,
                        accum["training"],
                        args.lambda_align,
                        effective_lambda_router,
                    )
                    sample_weight_rows.extend(weight_rows)
                except Exception:
                    log_error(output_dir, f"{protocol}_seed{seed}_{method}_{branch}_lambda{effective_lambda_router}")
                    continue

                val_detail = predict_adapter_mode(model, branch_feature(features["val"], branch), device)
                test_detail = predict_adapter_mode(model, branch_feature(features["test"], branch), device)
                branch_model_outputs[branch] = val_detail["logits"]
                branch_model_outputs_test[branch] = test_detail["logits"]
                branch_detail_val[branch] = val_detail
                branch_detail_test[branch] = test_detail
                trained_branch_models[branch] = model
                sample_weight_rows = sample_weight_rows

                align = alignment_row(protocol, seed, method, "learned_router", "test", branch, z_full["test"], test_detail["z_before"], test_detail["z_after"], tables["test"] == MISSING_COMPLETE)
                if align:
                    align["lambda_router"] = effective_lambda_router
                    accum["alignment"].append(align)
                if isinstance(model, MixtureFeatureAdapter):
                    val_complete_mask = tables["val"] == MISSING_COMPLETE
                    accum["router"].append(
                        router_stats_row(
                            protocol,
                            seed,
                            method,
                            branch,
                            "val",
                            branch_clusters[branch]["val"],
                            val_detail["alpha"],
                            "learned_router",
                            effective_lambda_router,
                            scope="complete_only",
                            mask=val_complete_mask,
                        )
                    )
                    accum["router"].append(
                        router_stats_row(
                            protocol,
                            seed,
                            method,
                            branch,
                            "test",
                            branch_clusters[branch]["test"],
                            test_detail["alpha"],
                            "learned_router",
                            effective_lambda_router,
                            scope="all",
                        )
                    )

            if not branch_model_outputs_test:
                continue
            val_logits = combined_for_method(branch_model_outputs, "val")
            test_logits = combined_for_method(branch_model_outputs_test, "test")
            threshold = select_protocol_thresholds(
                val_logits,
                labels["val"],
                tables["val"],
                args,
                protocol,
                seed,
                method,
                "learned_router",
                accum,
            )
            for _br, _detail in branch_detail_test.items():
                accum.setdefault("beta", []).extend(
                    beta_summary_rows(
                        args, protocol, seed, method, "test", _br, tables["test"], _detail, _detail["logits"], labels["test"], threshold
                    )
                )

            evaluate_and_record_method(
                args,
                protocol,
                seed,
                method,
                "learned_router",
                args.lambda_align,
                effective_lambda_router,
                threshold,
                val_logits,
                test_logits,
                labels,
                tables,
                full_logits,
                summary,
                accum,
                True,
                False,
                False,
            )
            macro = metrics_for_logits(test_logits, labels["test"], threshold, tables["test"])["macro_f1"]
            all_method_test_macros[(method, "learned_router", effective_lambda_router)] = macro
            if method == "static_feature_adapter":
                static_macro = macro
            if method == "wide_static_feature_adapter":
                wide_macro = macro
            static_ref = None
            for branch, model in trained_branch_models.items():
                if method == "static_feature_adapter":
                    static_ref = count_params(model, True)
                accum["params"].append(adapter_param_row(method, protocol, branch, seed, model, macro, static_ref))
            accum["sample_weight"].extend(sample_weight_rows)

            if method == "prototype_plain_hard":
                save_checkpoint(output_dir, protocol, seed, method, trained_branch_models, branch_centers, threshold, effective_lambda_router, args.lambda_align)
                if args.matched_oracle_eval:
                    for branch in branches:
                        generate_matched_oracle_rows(
                        args,
                        protocol,
                        seed,
                        branch,
                        trained_branch_models[branch],
                        branch_centers[branch],
                        branch_clusters[branch],
                        {split: branch_feature(features[split], branch) for split in ("train", "val", "test")},
                        z_full,
                        tables,
                        full_logits,
                        labels,
                        summary,
                        missing_threshold,
                        args.lambda_align,
                        effective_lambda_router,
                        {**accum, "device": device, "sample_ids": {split: features[split]["sample_ids"] for split in ("train", "val", "test")}},
                        output_dir,
                    )

    # Result post-processing.
    if static_macro is not None:
        for row in accum["results"]:
            if row["protocol"] == protocol and row["seed"] == seed:
                row["delta_vs_static"] = row["macro_f1"] - static_macro
                row["delta_vs_wide_static"] = row["macro_f1"] - wide_macro if wide_macro is not None else float("nan")
                row["delta_vs_missing_only"] = row["macro_f1"] - all_method_test_macros[("missing_only", "learned_router", 0.0)]
    accum["label_distribution"].extend(label_rows)
    accum["sampling_warnings"].extend(warnings_rows)


def write_outputs(args: argparse.Namespace, output_dir: Path, accum: Dict[str, List[Dict]], reference: Dict[str, pd.DataFrame]) -> None:
    results_df = write_csv(accum["results"], output_dir / "fix_results.csv")
    subgroup_df = write_csv(accum["subgroups"], output_dir / "subgroup_results.csv")
    missing_table_df = write_csv(accum["missing_tables"], output_dir / "missing_table_summary.csv")
    full_teacher_df = write_csv(accum["full_teacher"], output_dir / "full_teacher_results.csv")
    missing_base_df = write_csv(accum["missing_base"], output_dir / "missing_base_results.csv")
    params_df = write_csv(accum["params"], output_dir / "parameter_count_comparison.csv")
    clusters_df = write_csv(accum["clusters"], output_dir / "residual_cluster_summary.csv")
    router_df = write_csv(accum["router"], output_dir / "router_prediction_stats.csv")
    alignment_df = write_csv(accum["alignment"], output_dir / "residual_alignment_stats.csv")
    training_df = write_csv(accum["training"], output_dir / "training_log.csv")
    threshold_df = write_csv(accum["thresholds"], output_dir / "threshold_selection.csv")
    matched_df = write_csv(accum["matched_oracle"], output_dir / "matched_oracle_results.csv")
    per_sample_df = write_csv(accum["per_sample"], output_dir / "per_sample_router_diagnostics.csv")
    sample_weight_df = write_csv(accum["sample_weight"], output_dir / "sample_weight_audit.csv")
    beta_df = write_csv(accum.get("beta", []), output_dir / "beta_summary.csv")
    branch_threshold_df = write_csv(accum.get("branch_thresholds", []), output_dir / "branch_threshold_selection.csv")

    label_df = pd.DataFrame(accum["label_distribution"])
    warnings_df = pd.DataFrame(accum["sampling_warnings"])
    if args.complete_sampling == "stratified_primary_label":
        label_df.to_csv(output_dir / "stratified_missing_table_summary.csv", index=False)
        label_df.to_csv(output_dir / "label_distribution_audit.csv", index=False)
    else:
        label_df.to_csv(output_dir / "label_distribution_audit.csv", index=False)

    if not results_df.empty:
        seed_summary = (
            results_df[results_df["routing_mode"] == "learned_router"]
            .groupby(["protocol", "sampling_mode", "method", "deployable"])["macro_f1"]
            .agg(["mean", "std", "count", "max"])
            .reset_index()
            .rename(columns={"mean": "macro_f1_mean", "std": "macro_f1_std", "count": "n_seeds", "max": "macro_f1_best"})
        )
        seed_summary.to_csv(output_dir / "seed_summary.csv", index=False)
        seed_summary.to_csv(output_dir / "safe_mixture_seed_summary.csv", index=False)
    else:
        seed_summary = pd.DataFrame()
        seed_summary.to_csv(output_dir / "seed_summary.csv", index=False)
        seed_summary.to_csv(output_dir / "safe_mixture_seed_summary.csv", index=False)

    # Weak router summary.
    weak_rows = []
    reference_results = reference["mora_aligned_results.csv"]
    reference_router = reference["router_prediction_stats.csv"]
    if not results_df.empty:
        proto_results = results_df[(results_df["method"] == "prototype_plain_hard") & (results_df["routing_mode"] == "learned_router")]
        if router_df.empty or "method" not in router_df.columns:
            router_learned = pd.DataFrame()
        else:
            router_learned = router_df[
                (router_df["method"] == "prototype_plain_hard")
                & (router_df["split"] == "val")
                & (router_df["phase"] == "learned_router")
                & (router_df["scope"] == "complete_only")
            ]
        for row in proto_results.itertuples(index=False):
            router_sub = router_learned[
                (router_learned["protocol"] == row.protocol)
                & (router_learned["seed"] == row.seed)
                & np.isclose(router_learned["lambda_router"], row.lambda_router)
            ]
            entropy = safe_value(router_sub["router_entropy_mean"].mean()) if not router_sub.empty else float("nan")
            router_acc = safe_value(router_sub["router_accuracy"].mean()) if not router_sub.empty else float("nan")
            old_macro = load_reference_macro(reference_results, row.protocol, int(row.seed), "prototype_plain_hard")
            mixture_macro = load_reference_macro(reference_results, row.protocol, int(row.seed), "mixture_feature_adapter_unsupervised")
            static_macro = load_reference_macro(reference_results, row.protocol, int(row.seed), "static_feature_adapter")
            wide_macro = load_reference_macro(reference_results, row.protocol, int(row.seed), "wide_static_feature_adapter")
            weak_rows.append(
                {
                    "seed": int(row.seed),
                    "protocol": row.protocol,
                    "sampling_mode": row.sampling_mode,
                    "lambda_router": float(row.lambda_router),
                    "lambda_align": float(row.lambda_align),
                    "macro_f1": float(row.macro_f1),
                    "micro_f1": float(row.micro_f1),
                    "sample_f1": float(row.sample_f1),
                    "bce_loss": float(row.bce_loss),
                    "alpha_entropy": entropy,
                    "alpha_max_mean": float(router_sub["alpha_max_mean"].mean()) if not router_sub.empty else float("nan"),
                    "router_accuracy_complete_val": router_acc,
                    "delta_vs_old_lambda_003": float(row.macro_f1 - old_macro) if not math.isnan(old_macro) else float("nan"),
                    "delta_vs_mixture": float(row.macro_f1 - mixture_macro) if not math.isnan(mixture_macro) else float("nan"),
                    "delta_vs_static": float(row.macro_f1 - static_macro) if not math.isnan(static_macro) else float("nan"),
                    "delta_vs_wide_static": float(row.macro_f1 - wide_macro) if not math.isnan(wide_macro) else float("nan"),
                }
            )
    # Add reference lambda 0.03 rows for comparison.
    if not reference_results.empty:
        ref_proto = reference_results[reference_results["method"] == "prototype_plain_hard"]
        for row in ref_proto.itertuples(index=False):
            if row.protocol not in args.protocols or args.complete_sampling != "random":
                continue
            entropy, router_acc = load_reference_router_stats(reference_router, row.protocol, int(row.seed), "prototype_plain_hard")
            mixture_macro = load_reference_macro(reference_results, row.protocol, int(row.seed), "mixture_feature_adapter_unsupervised")
            static_macro = load_reference_macro(reference_results, row.protocol, int(row.seed), "static_feature_adapter")
            wide_macro = load_reference_macro(reference_results, row.protocol, int(row.seed), "wide_static_feature_adapter")
            weak_rows.append(
                {
                    "seed": int(row.seed),
                    "protocol": row.protocol,
                    "sampling_mode": "random_reference",
                    "lambda_router": 0.03,
                    "lambda_align": args.lambda_align,
                    "macro_f1": float(row.macro_f1),
                    "micro_f1": float(row.micro_f1),
                    "sample_f1": float(row.sample_f1),
                    "bce_loss": float(row.bce_loss),
                    "alpha_entropy": entropy,
                    "alpha_max_mean": float("nan"),
                    "router_accuracy_complete_val": router_acc,
                    "delta_vs_old_lambda_003": 0.0,
                    "delta_vs_mixture": float(row.macro_f1 - mixture_macro) if not math.isnan(mixture_macro) else float("nan"),
                    "delta_vs_static": float(row.macro_f1 - static_macro) if not math.isnan(static_macro) else float("nan"),
                    "delta_vs_wide_static": float(row.macro_f1 - wide_macro) if not math.isnan(wide_macro) else float("nan"),
                }
            )
    weak_df = write_csv(weak_rows, output_dir / "weak_router_loss_results.csv")

    if args.complete_sampling == "stratified_primary_label":
        strat_rows = []
        random_ref = reference["mora_aligned_results.csv"]
        deployable = results_df[results_df["routing_mode"] == "learned_router"]
        for row in deployable.itertuples(index=False):
            random_macro = load_reference_macro(random_ref, row.protocol, int(row.seed), row.method)
            mixture_macro = load_reference_macro(random_ref, row.protocol, int(row.seed), "mixture_feature_adapter_unsupervised")
            static_macro = load_reference_macro(random_ref, row.protocol, int(row.seed), "static_feature_adapter")
            wide_macro = load_reference_macro(random_ref, row.protocol, int(row.seed), "wide_static_feature_adapter")
            missing_macro = load_reference_macro(random_ref, row.protocol, int(row.seed), "missing_only")
            strat_rows.append(
                {
                    "seed": int(row.seed),
                    "sampling_mode": row.sampling_mode,
                    "method": row.method,
                    "macro_f1": float(row.macro_f1),
                    "micro_f1": float(row.micro_f1),
                    "sample_f1": float(row.sample_f1),
                    "bce_loss": float(row.bce_loss),
                    "delta_vs_random_sampling_same_method": float(row.macro_f1 - random_macro) if not math.isnan(random_macro) else float("nan"),
                    "delta_vs_missing_only": float(row.macro_f1 - missing_macro) if not math.isnan(missing_macro) else float("nan"),
                    "delta_vs_static": float(row.macro_f1 - static_macro) if not math.isnan(static_macro) else float("nan"),
                    "delta_vs_wide_static": float(row.macro_f1 - wide_macro) if not math.isnan(wide_macro) else float("nan"),
                    "delta_vs_mixture": float(row.macro_f1 - mixture_macro) if not math.isnan(mixture_macro) else float("nan"),
                }
            )
        write_csv(strat_rows, output_dir / "stratified_experiment_results.csv")

    make_summary(
        args,
        output_dir,
        results_df,
        matched_df,
        weak_df,
        sample_weight_df,
        seed_summary,
        reference["seed_summary.csv"],
        label_df,
        warnings_df,
        beta_df,
    )


def safe_value(value) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def mean_std_text(df: pd.DataFrame, filter_mask: pd.Series) -> str:
    sub = df[filter_mask]
    if sub.empty:
        return "NA"
    mean = sub["macro_f1"].mean()
    std = sub["macro_f1"].std(ddof=1)
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def make_summary(
    args: argparse.Namespace,
    output_dir: Path,
    results_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    weak_df: pd.DataFrame,
    sample_weight_df: pd.DataFrame,
    seed_summary: pd.DataFrame,
    reference_seed_summary: pd.DataFrame,
    label_df: pd.DataFrame,
    warnings_df: pd.DataFrame,
    beta_df: pd.DataFrame,
) -> None:
    if "routing_mode" not in matched_df.columns:
        matched_df = pd.DataFrame(columns=["protocol", "routing_mode", "macro_f1", "seed"])
    if "protocol" not in weak_df.columns:
        weak_df = pd.DataFrame(columns=["protocol", "sampling_mode", "lambda_router", "macro_f1", "alpha_entropy", "router_accuracy_complete_val"])
    if "sample_type" not in sample_weight_df.columns:
        sample_weight_df = pd.DataFrame(columns=["sample_type", "relative_to_incomplete"])
    lines = [
        "# RUN SUMMARY",
        "",
        "## 1. 本輪修正目標",
        "這輪不是新方法，而是修 prototype 的三個核心問題：matched oracle、weak router supervision、以及 stratified complete sampling。",
        "",
        "## 2. Matched oracle 結果",
    ]
    matched_sub = matched_df[matched_df["protocol"] == "image_missing_70"] if not matched_df.empty else matched_df.copy()
    learned = matched_sub[matched_sub["routing_mode"] == "learned_router"]
    hard = matched_sub[matched_sub["routing_mode"] == "oracle_hard_same_checkpoint"]
    soft = matched_sub[matched_sub["routing_mode"] == "oracle_soft_same_checkpoint"]
    lines.append(f"- learned router macro-F1: {mean_std_text(learned, learned['routing_mode'] == 'learned_router') if not learned.empty else 'NA'}")
    lines.append(f"- oracle hard same checkpoint macro-F1: {mean_std_text(hard, hard['routing_mode'] == 'oracle_hard_same_checkpoint') if not hard.empty else 'NA'}")
    if not soft.empty:
        best_soft = soft.sort_values("macro_f1", ascending=False).groupby("seed").head(1)
        lines.append(f"- oracle soft same checkpoint macro-F1: {mean_std_text(best_soft, best_soft['routing_mode'] == 'oracle_soft_same_checkpoint')}")
        learned_mean = learned["macro_f1"].mean() if not learned.empty else float("nan")
        soft_mean = best_soft["macro_f1"].mean()
        if not math.isnan(learned_mean):
            lines.append(f"- oracle soft vs learned delta: {soft_mean - learned_mean:+.4f}")
    if not hard.empty and not learned.empty:
        hard_mean = hard["macro_f1"].mean()
        learned_mean = learned["macro_f1"].mean()
        lines.append(f"- oracle hard vs learned delta: {hard_mean - learned_mean:+.4f}")
        if hard_mean <= learned_mean and (soft.empty or soft["macro_f1"].max() <= learned_mean):
            lines.append("- 判斷：matched oracle 沒有明顯高於 learned router，代表 residual cluster target 可能不是分類最佳 target。")
        else:
            lines.append("- 判斷：matched oracle 高於 learned router，代表 router learning 仍有改善空間。")
    lines.extend(["", "## 3. Weak router loss 結果"])
    weak_img = weak_df[weak_df["protocol"] == "image_missing_70"] if not weak_df.empty else pd.DataFrame()
    if weak_img.empty:
        lines.append("- 無 weak router 結果。")
    else:
        for lam in [0.001, 0.003, 0.01, 0.03]:
            sub = weak_img[np.isclose(weak_img["lambda_router"], lam)]
            if sub.empty:
                continue
            macro_mean = sub["macro_f1"].mean()
            macro_std = sub["macro_f1"].std(ddof=1)
            ent = sub["alpha_entropy"].mean()
            acc = sub["router_accuracy_complete_val"].mean()
            std_text = "NA" if pd.isna(macro_std) else f"{macro_std:.4f}"
            lines.append(f"- lambda_router={lam:.3f}: macro-F1={macro_mean:.4f} +/- {std_text}, alpha_entropy={ent:.4f}, router_acc={acc:.4f}")
        learned_only = weak_img[weak_img["sampling_mode"] != "random_reference"]
        if not learned_only.empty:
            best_row = learned_only.sort_values("macro_f1", ascending=False).iloc[0]
            lines.append(f"- best weak router lambda: {best_row['lambda_router']:.3f}, macro-F1={best_row['macro_f1']:.4f}")
            ref = weak_img[np.isclose(weak_img["lambda_router"], 0.03)]
            if not ref.empty:
                lines.append(f"- delta vs old 0.03: {best_row['macro_f1'] - ref['macro_f1'].mean():+.4f}")
                if best_row["macro_f1"] > ref["macro_f1"].mean():
                    lines.append("- 判斷：weak router supervision 較好，表示原本 hard CE router supervision 可能太強。")
                else:
                    lines.append("- 判斷：weak router supervision 沒有穩定優於舊設定。")
    lines.extend(["", "## 4. Stratified complete sampling 結果"])
    if args.complete_sampling != "stratified_primary_label":
        lines.append("- 這個輸出目錄沒有跑 stratified complete sampling。")
    else:
        strat_path = output_dir / "stratified_experiment_results.csv"
        strat_df = pd.read_csv(strat_path) if strat_path.exists() else pd.DataFrame()
        if strat_df.empty:
            lines.append("- 沒有 stratified 結果。")
        else:
            for method in ["missing_only", "static_feature_adapter", "wide_static_feature_adapter", "mixture_feature_adapter_unsupervised", "prototype_plain_hard"]:
                sub = strat_df[strat_df["method"] == method]
                if sub.empty:
                    continue
                delta = sub["delta_vs_random_sampling_same_method"].mean()
                lines.append(f"- {method}: stratified macro-F1 mean={sub['macro_f1'].mean():.4f}, delta vs random={delta:+.4f}")
            proto = strat_df[strat_df["method"] == "prototype_plain_hard"]
            if not proto.empty:
                lines.append(f"- prototype std under stratified={proto['macro_f1'].std(ddof=1):.4f}")
                if proto["delta_vs_random_sampling_same_method"].mean() > 0:
                    lines.append("- 判斷：stratified complete sampling 對 prototype 有正面幫助。")
                else:
                    lines.append("- 判斷：stratified complete sampling 沒有明顯改善 prototype。")
    lines.extend(["", "## 5. Complete sample weight audit"])
    if sample_weight_df.empty:
        lines.append("- 無 sample weight audit。")
    else:
        complete = sample_weight_df[sample_weight_df["sample_type"] == "complete"]
        incomplete = sample_weight_df[sample_weight_df["sample_type"] == "incomplete"]
        if not complete.empty:
            ratio = complete["relative_to_incomplete"].mean()
            lines.append(f"- complete relative_to_incomplete mean={ratio:.4f}")
            if ratio > 1.2:
                lines.append("- 判斷：complete samples 仍然因多重 loss 承擔較高 effective weight。")
            else:
                lines.append("- 判斷：loss normalization 已經把 complete / incomplete 的總體 loss scale 拉得更接近。")
    lines.extend(["", "## 6. 對 prototype 的最終判斷"])
    case = "Case D"
    if args.complete_sampling == "stratified_primary_label":
        strat_path = output_dir / "stratified_experiment_results.csv"
        strat_df = pd.read_csv(strat_path) if strat_path.exists() else pd.DataFrame()
        if not strat_df.empty:
            proto_delta = strat_df[strat_df["method"] == "prototype_plain_hard"]["delta_vs_random_sampling_same_method"].mean()
            if proto_delta > 0.01:
                case = "Case C"
    if not weak_df.empty and not matched_df.empty:
        learned_mean = learned["macro_f1"].mean() if not learned.empty else float("nan")
        hard_mean = hard["macro_f1"].mean() if not hard.empty else float("nan")
        weak_best = weak_img[weak_img["sampling_mode"] != "random_reference"]["macro_f1"].max() if not weak_img.empty else float("nan")
        ref_old = weak_img[np.isclose(weak_img["lambda_router"], 0.03)]["macro_f1"].mean() if not weak_img.empty else float("nan")
        if not math.isnan(hard_mean) and not math.isnan(learned_mean) and hard_mean > learned_mean and not math.isnan(weak_best) and not math.isnan(ref_old) and weak_best > ref_old:
            case = "Case A"
        elif not math.isnan(weak_best) and not math.isnan(ref_old) and weak_best > ref_old:
            case = "Case B"
    lines.append(f"- {case}")
    if case == "Case A":
        lines.append("- matched oracle > learned router 且 weak router 改善，代表 prototype target 有空間，主要問題在 router learning。")
    elif case == "Case B":
        lines.append("- matched oracle 沒明顯更高，但 weak router 有幫助，代表 prototype cluster 更像 weak auxiliary regularizer。")
    elif case == "Case C":
        lines.append("- stratified sampling 明顯幫到 prototype，代表 complete sample coverage 是主要瓶頸。")
    else:
        lines.append("- 目前沒有足夠證據把 prototype 升級成主方法，sample-aware mixture 仍是較穩主線。")
    lines.extend(["", "## 7. Thesis 建議"])
    if case in {"Case A", "Case C"}:
        lines.append("- prototype 可以繼續保留，但目前仍建議先寫成 mixture 主線、prototype 作 guided/auxiliary 分支。")
    else:
        lines.append("- 目前更保守的寫法是 mixture 當主方法，prototype 當 auxiliary / interpretability。")
    lines.append("- 在 image_missing_70 修穩之前，不建議先擴 text_missing / both_70。")
    lines.append("- 在 matched oracle 與 weak-router 修正都成立之前，不建議直接接 MoRA backbone。")
    if args.complete_sampling == "stratified_primary_label" and not warnings_df.empty:
        lines.extend(["", "## 8. Stratified sampling warnings"])
        for row in warnings_df.itertuples(index=False):
            lines.append(f"- {row.protocol} seed={row.seed} split={row.split}: {row.warning}")
    # The old prototype-oriented summary above is kept for backward compatibility,
    # but final thesis experiments now focus on safe_mixture_* methods.
    # Overwrite RUN_SUMMARY.md with a final-method summary when safe-mixture rows exist.
    safe_methods = [
        "missing_only",
        "static_feature_adapter",
        "wide_static_feature_adapter",
        "mixture_feature_adapter_unsupervised",
        "safe_mixture_beta_zero",
        "safe_mixture_beta_one",
        "safe_mixture_beta_learned",
        "safe_mixture_beta_biased",
        "safe_mixture_feature_adapter",
    ]
    safe_results = results_df[results_df["method"].isin(safe_methods)] if not results_df.empty and "method" in results_df.columns else pd.DataFrame()
    if not safe_results.empty:
        safe_lines = [
            "# Safe Mixture Final RUN SUMMARY",
            "",
            "## 1. 本輪目標",
            "這輪只看最終版 Missing-Type Conditional Safe Residual Mixture：full teacher 只作 training-time residual supervision，不再作 final classifier。",
            "推論時 complete samples 使用 missing-type branches 的 ensemble；incomplete samples 使用對應 available-modality branch。",
            "重點是確認 static fallback、ordinary mixture、以及 beta-controlled safe mixture 在 image_missing_70 / text_missing_70 / both_70 的表現。",
            "",
            "## 2. 主結果 macro-F1 mean/std",
        ]
        learned_safe = safe_results[safe_results["routing_mode"] == "learned_router"] if "routing_mode" in safe_results.columns else safe_results
        if learned_safe.empty:
            safe_lines.append("- 沒有 learned_router 結果。")
        else:
            for protocol in args.protocols:
                safe_lines.append("")
                safe_lines.append(f"### {protocol}")
                subp = learned_safe[learned_safe["protocol"] == protocol]
                if subp.empty:
                    safe_lines.append("- 無結果。")
                    continue
                summary = (
                    subp.groupby("method")["macro_f1"]
                    .agg(["mean", "std", "count", "max"])
                    .reset_index()
                    .sort_values("mean", ascending=False)
                )
                for row in summary.itertuples(index=False):
                    std_text = "NA" if pd.isna(row.std) else f"{row.std:.4f}"
                    safe_lines.append(f"- {row.method}: {row.mean:.4f} +/- {std_text} (n={int(row.count)}, best={row.max:.4f})")
        safe_lines.extend(["", "## 3. Beta 行為"])
        if beta_df.empty or "beta_mean" not in beta_df.columns:
            safe_lines.append("- 無 beta_summary，請確認是否有跑 safe_mixture_* method。")
        else:
            for protocol in args.protocols:
                safe_lines.append("")
                safe_lines.append(f"### {protocol}")
                subb = beta_df[beta_df["protocol"] == protocol] if "protocol" in beta_df.columns else pd.DataFrame()
                if subb.empty:
                    safe_lines.append("- 無 beta diagnostics。")
                    continue
                group_cols = [col for col in ["method", "subgroup", "branch"] if col in subb.columns]
                beta_summary = subb.groupby(group_cols)["beta_mean"].mean().reset_index() if group_cols else pd.DataFrame()
                for row in beta_summary.itertuples(index=False):
                    parts = [str(getattr(row, col)) for col in group_cols]
                    safe_lines.append(f"- {' / '.join(parts)}: beta_mean={row.beta_mean:.4f}")
        safe_lines.extend(["", "## 4. 判讀重點"])
        safe_lines.append("- beta_zero 應該接近 static fallback；beta_one 是強制加 mixture；learned/biased beta 才是正式 safe controller。")
        safe_lines.append("- complete samples 不再使用 full teacher logits，而是使用 branch ensemble，避免 30% complete teacher 在 test complete 上不穩。")
        safe_lines.append("- 若 text_missing 中 beta_biased 接近 static 且不被 beta_one 拖垮，表示 safe fallback 有效。")
        safe_lines.append("- 若 both_70 中 safe_mixture_beta_biased 比 ordinary mixture 穩，才可以把 safe mixture 當通用 missing-modality framework。")
        safe_lines.extend(["", "## 5. 輸出檔案"])
        for name in [
            "fix_results.csv",
            "safe_mixture_seed_summary.csv",
            "beta_summary.csv",
            "subgroup_results.csv",
            "branch_threshold_selection.csv",
            "per_sample_router_diagnostics.csv",
            "sample_weight_audit.csv",
            "threshold_selection.csv",
            "training_log.csv",
            "config.json",
            "errors.log",
        ]:
            safe_lines.append(f"- {output_dir / name}")
        (output_dir / "RUN_SUMMARY.md").write_text("\n".join(safe_lines) + "\n", encoding="utf-8")
    else:
        (output_dir / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(resolve_path(args.output_dir))
    (output_dir / "errors.log").write_text("", encoding="utf-8")
    device = device_from_args(args)
    reference_dir = resolve_path(args.reference_results_dir)
    reference = load_reference_outputs(reference_dir)
    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["note"] = (
        "Full teacher is training-only residual supervision. Complete-sample evaluation defaults to "
        "branch_ensemble, so deployable methods do not use full-teacher logits."
    )
    save_json(config, output_dir / "config.json")

    features, labels, split_meta, class_names, feature_meta = load_mmimdb_payload(resolve_path(args.feature_dir), resolve_path(args.metadata_csv))
    save_json(
        {
            "feature_metadata": feature_meta,
            "class_names": class_names,
            "splits": {
                split: {
                    "image_shape": list(features[split]["image"].shape),
                    "text_shape": list(features[split]["text"].shape),
                    "label_shape": list(labels[split].shape),
                    "n_samples": int(len(features[split]["sample_ids"])),
                }
                for split in ("train", "val", "test")
            },
        },
        output_dir / "audit.json",
    )

    accum = {
        "results": [],
        "subgroups": [],
        "missing_tables": [],
        "full_teacher": [],
        "missing_base": [],
        "params": [],
        "clusters": [],
        "router": [],
        "alignment": [],
        "training": [],
        "thresholds": [],
        "matched_oracle": [],
        "per_sample": [],
        "sample_weight": [],
        "label_distribution": [],
        "sampling_warnings": [],
        "beta": [],
        "branch_thresholds": [],
    }
    shared = {
        "features": features,
        "labels": labels,
        "split_meta": split_meta,
        "class_names": class_names,
        "feature_meta": feature_meta,
        "device": device,
        "primary_labels": primary_labels_for_split(split_meta, labels),
    }
    for protocol in args.protocols:
        for seed in args.seeds:
            try:
                run_protocol_seed(args, protocol, seed, shared, output_dir, reference, accum)
            except Exception:
                log_error(output_dir, f"{protocol}_seed{seed}")
    write_outputs(args, output_dir, accum, reference)

    print("# Safe Mixture Final Summary")
    print("")
    seed_summary_path = output_dir / "safe_mixture_seed_summary.csv"
    seed_df = safe_read_csv(seed_summary_path)
    if seed_df.empty:
        seed_df = safe_read_csv(output_dir / "seed_summary.csv")
    if seed_df.empty or "method" not in seed_df.columns:
        print("No seed summary available.")
    else:
        for protocol in args.protocols:
            print(f"## {protocol}")
            sub = seed_df[seed_df["protocol"] == protocol] if "protocol" in seed_df.columns else pd.DataFrame()
            if sub.empty:
                print("- no results")
                continue
            sub = sub.sort_values("macro_f1_mean", ascending=False) if "macro_f1_mean" in sub.columns else sub
            for row in sub.itertuples(index=False):
                mean = getattr(row, "macro_f1_mean", float("nan"))
                std = getattr(row, "macro_f1_std", float("nan"))
                n = getattr(row, "n_seeds", getattr(row, "count", ""))
                std_text = "NA" if pd.isna(std) else f"{std:.4f}"
                print(f"- {row.method}: {mean:.4f} +/- {std_text} (n={n})")
            print("")

    beta_df = safe_read_csv(output_dir / "beta_summary.csv")
    print("## Beta behavior")
    if beta_df.empty or "beta_mean" not in beta_df.columns:
        print("- no beta diagnostics")
    else:
        for subgroup in ["image_missing", "text_missing", "complete"]:
            sub = beta_df[beta_df["subgroup"] == subgroup] if "subgroup" in beta_df.columns else pd.DataFrame()
            if sub.empty:
                continue
            print(f"{subgroup}_beta_mean: {sub['beta_mean'].mean():.4f}")
    print("")
    print("## Main conclusion")
    print("一句話：full teacher 已限制為 training-only supervision；請以 branch-ensemble complete evaluation 下 safe_mixture_beta_biased 是否超過 static / ordinary mixture 判斷方法是否成立。")
    print("")
    print("## Files")
    for name in [
        "RUN_SUMMARY.md",
        "fix_results.csv",
        "safe_mixture_seed_summary.csv",
        "beta_summary.csv",
        "subgroup_results.csv",
        "branch_threshold_selection.csv",
        "per_sample_router_diagnostics.csv",
        "sample_weight_audit.csv",
        "threshold_selection.csv",
        "training_log.csv",
        "config.json",
        "errors.log",
    ]:
        print(f"- {output_dir / name}")


if __name__ == "__main__":
    main()
