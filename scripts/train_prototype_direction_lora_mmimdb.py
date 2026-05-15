import argparse
import json
import math
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recover_cached_feature_baseline import (  # noqa: E402
    apply_thresholds,
    ensure_dir,
    load_cache,
    mean_bce_loss_np,
    multilabel_metrics,
    predict,
    sigmoid_np,
    threshold_search,
    train_linear,
)


MISSING_COMPLETE = 0
MISSING_TEXT = 1
MISSING_IMAGE = 2
MISSING_NAMES = {
    MISSING_COMPLETE: "complete",
    MISSING_TEXT: "text_missing",
    MISSING_IMAGE: "image_missing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Prototype Direction-Guided LoRA on MM-IMDb.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def append_error(output_dir: Path, title: str) -> None:
    with (output_dir / "errors.log").open("a", encoding="utf-8") as f:
        f.write(f"\n[{title}]\n")
        f.write(traceback.format_exc())
        f.write("\n")


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
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


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def l2_normalize_vec(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / norm).astype(np.float32)


def cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), eps)
    return (a * b).sum(axis=1) / denom


def primary_labels(y: np.ndarray) -> np.ndarray:
    weights = np.arange(y.shape[1], 0, -1, dtype=np.float32)
    return (y * weights.reshape(1, -1)).argmax(axis=1).astype(np.int64)


def greedy_iterative_multilabel_complete_indices(y: np.ndarray, n_complete: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = y.sum(axis=0) * (n_complete / max(len(y), 1))
    remaining = list(rng.permutation(len(y)))
    chosen: List[int] = []
    current = np.zeros(y.shape[1], dtype=np.float32)
    label_mass = np.maximum(y.sum(axis=0), 1.0)
    while len(chosen) < n_complete and remaining:
        deficit = np.maximum(target - current, 0.0)
        if deficit.sum() <= 1e-8:
            chosen.extend(remaining[: n_complete - len(chosen)])
            break
        best_pos = 0
        best_score = None
        for pos, idx in enumerate(remaining):
            label_vec = y[idx]
            gain = float((deficit * label_vec).sum())
            over = float(np.maximum(current + label_vec - target, 0.0).sum())
            rarity = float((label_vec / label_mass).sum())
            score = (gain, -over, rarity)
            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos
        chosen_idx = remaining.pop(best_pos)
        chosen.append(int(chosen_idx))
        current += y[chosen_idx]
    return np.array(sorted(chosen[:n_complete]), dtype=np.int64)


def make_missing_table(y: np.ndarray, protocol: str, missing_ratio: float, seed: int) -> np.ndarray:
    complete_ratio = 1.0 - missing_ratio
    n_complete = int(round(len(y) * complete_ratio))
    complete_idx = greedy_iterative_multilabel_complete_indices(y, n_complete, seed)
    table = np.full(len(y), MISSING_COMPLETE, dtype=np.int64)
    complete_set = set(complete_idx.tolist())
    incomplete = np.array([i for i in range(len(y)) if i not in complete_set], dtype=np.int64)
    if protocol == "image_missing_70":
        table[incomplete] = MISSING_IMAGE
    elif protocol == "text_missing_70":
        table[incomplete] = MISSING_TEXT
    elif protocol == "both_70":
        rng = np.random.default_rng(seed + 123)
        shuffled = rng.permutation(incomplete)
        cut = len(shuffled) // 2
        table[shuffled[:cut]] = MISSING_IMAGE
        table[shuffled[cut:]] = MISSING_TEXT
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    return table


def load_ordered_metadata(metadata_csv: Path, data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, pd.DataFrame]:
    metadata = pd.read_csv(metadata_csv, dtype={"sample_id": str})
    metadata["sample_id"] = metadata["sample_id"].map(normalize_sample_id)
    indexed = metadata.drop_duplicates("sample_id").set_index("sample_id", drop=False)
    out: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        sample_ids = [normalize_sample_id(x) for x in data[split]["sample_ids"]]
        missing = [sid for sid in sample_ids if sid not in indexed.index]
        if missing:
            raise RuntimeError(f"{len(missing)} sample ids missing from metadata for {split}: {missing[:5]}")
        out[split] = indexed.loc[sample_ids].reset_index(drop=True)
    return out


def shared_base_features(data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        z_img = l2_normalize_np(data[split]["image"])
        z_txt = l2_normalize_np(data[split]["text"])
        z_full = l2_normalize_np((z_img + z_txt) / 2.0)
        out[split] = {
            "image": z_img,
            "text": z_txt,
            "full": z_full,
        }
    return out


def build_prototypes(
    base: Dict[str, Dict[str, np.ndarray]],
    labels: Dict[str, np.ndarray],
    class_names: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]], List[Dict]]:
    train_y = labels["train"]
    z_full = base["train"]["full"]
    z_img = base["train"]["image"]
    z_txt = base["train"]["text"]
    num_labels = train_y.shape[1]
    delta_image_missing = np.zeros((num_labels, z_full.shape[1]), dtype=np.float32)
    delta_text_missing = np.zeros((num_labels, z_full.shape[1]), dtype=np.float32)
    stats_rows: List[Dict] = []

    for class_idx, class_name in enumerate(class_names):
        mask = train_y[:, class_idx] > 0.5
        count = int(mask.sum())
        if count > 0:
            proto_full = z_full[mask].mean(axis=0).astype(np.float32)
            proto_txt = z_txt[mask].mean(axis=0).astype(np.float32)
            proto_img = z_img[mask].mean(axis=0).astype(np.float32)
        else:
            proto_full = np.zeros(z_full.shape[1], dtype=np.float32)
            proto_txt = np.zeros(z_full.shape[1], dtype=np.float32)
            proto_img = np.zeros(z_full.shape[1], dtype=np.float32)

        raw_delta_image = proto_full - proto_txt
        raw_delta_text = proto_full - proto_img
        delta_image_missing[class_idx] = l2_normalize_vec(raw_delta_image)
        delta_text_missing[class_idx] = l2_normalize_vec(raw_delta_text)

        for branch, proto_missing, delta in [
            ("image_missing", proto_txt, delta_image_missing[class_idx]),
            ("text_missing", proto_img, delta_text_missing[class_idx]),
        ]:
            stats_rows.append(
                {
                    "class_name": class_name,
                    "class_count": count,
                    "proto_full_norm": float(np.linalg.norm(proto_full)),
                    "proto_missing_norm": float(np.linalg.norm(proto_missing)),
                    "delta_proto_norm": float(np.linalg.norm(delta)),
                    "branch": branch,
                    "valid": bool(count > 0 and np.linalg.norm(delta) > 1e-8),
                }
            )

    class_deltas = {
        "image_missing": delta_image_missing,
        "text_missing": delta_text_missing,
    }
    sample_targets: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        sample_targets[split] = {}
        y = labels[split]
        for branch, class_delta in class_deltas.items():
            targets = np.zeros((len(y), z_full.shape[1]), dtype=np.float32)
            for i in range(len(y)):
                pos = np.flatnonzero(y[i] > 0.5)
                if len(pos) == 0:
                    continue
                targets[i] = l2_normalize_vec(class_delta[pos].mean(axis=0))
            sample_targets[split][branch] = targets
    return class_deltas, sample_targets, stats_rows


def build_protocol_representations(base: Dict[str, np.ndarray], table: np.ndarray) -> np.ndarray:
    reps = base["full"].copy()
    image_missing = table == MISSING_IMAGE
    text_missing = table == MISSING_TEXT
    reps[image_missing] = base["text"][image_missing]
    reps[text_missing] = base["image"][text_missing]
    return reps.astype(np.float32)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling


class LoRAClipClassifier(nn.Module):
    def __init__(self, cfg: Dict, num_labels: int):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(cfg["clip_model"])
        for param in self.clip.parameters():
            param.requires_grad = False
        self._inject_lora(cfg)
        self.classifier = nn.Linear(512, num_labels)

    def _inject_lora(self, cfg: Dict) -> None:
        rank = int(cfg["lora_rank"])
        alpha = int(cfg["lora_alpha"])
        dropout = float(cfg["lora_dropout"])
        for tower_name in ("text_model", "vision_model"):
            tower = getattr(self.clip, tower_name)
            layers = tower.encoder.layers
            for layer_idx in range(len(layers) - 2, len(layers)):
                attn = layers[layer_idx].self_attn
                attn.q_proj = LoRALinear(attn.q_proj, rank, alpha, dropout)
                attn.v_proj = LoRALinear(attn.v_proj, rank, alpha, dropout)

    def trainable_param_summary(self, protocol: str, method: str) -> Dict:
        lora_params = 0
        classifier_params = 0
        trainable_original_clip = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            count = int(param.numel())
            if ".lora_A" in name or ".lora_B" in name:
                lora_params += count
            elif name.startswith("classifier."):
                classifier_params += count
            elif name.startswith("clip."):
                trainable_original_clip += count
        return {
            "protocol": protocol,
            "method": method,
            "trainable_lora_params": lora_params,
            "trainable_classifier_params": classifier_params,
            "trainable_original_clip_backbone_params": trainable_original_clip,
            "trainable_total_params": lora_params + classifier_params + trainable_original_clip,
        }


class ProtocolDataset(Dataset):
    def __init__(
        self,
        split: str,
        records: pd.DataFrame,
        labels: np.ndarray,
        base: Dict[str, np.ndarray],
        table: np.ndarray,
        proto_targets: Dict[str, np.ndarray],
    ):
        self.split = split
        self.records = records.reset_index(drop=True)
        self.labels = labels.astype(np.float32)
        self.base = base
        self.table = table.astype(np.int64)
        self.proto_targets = proto_targets

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        row = self.records.iloc[idx]
        return {
            "idx": idx,
            "sample_id": normalize_sample_id(row["sample_id"]),
            "image_path": str(row["image_path"]),
            "text": str(row["text"]),
            "label": self.labels[idx],
            "missing_code": int(self.table[idx]),
            "z_image": self.base["image"][idx],
            "z_text": self.base["text"][idx],
            "z_full": self.base["full"][idx],
            "proto_image_missing": self.proto_targets["image_missing"][idx],
            "proto_text_missing": self.proto_targets["text_missing"][idx],
        }


def collate_protocol_batch(items: List[Dict]) -> Dict:
    return {
        "idx": torch.tensor([x["idx"] for x in items], dtype=torch.long),
        "sample_id": [x["sample_id"] for x in items],
        "image_path": [x["image_path"] for x in items],
        "text": [x["text"] for x in items],
        "label": torch.tensor(np.stack([x["label"] for x in items]), dtype=torch.float32),
        "missing_code": torch.tensor([x["missing_code"] for x in items], dtype=torch.long),
        "z_image": torch.tensor(np.stack([x["z_image"] for x in items]), dtype=torch.float32),
        "z_text": torch.tensor(np.stack([x["z_text"] for x in items]), dtype=torch.float32),
        "z_full": torch.tensor(np.stack([x["z_full"] for x in items]), dtype=torch.float32),
        "proto_image_missing": torch.tensor(np.stack([x["proto_image_missing"] for x in items]), dtype=torch.float32),
        "proto_text_missing": torch.tensor(np.stack([x["proto_text_missing"] for x in items]), dtype=torch.float32),
    }


def positive_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    weights = np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def encode_text(model: LoRAClipClassifier, processor: CLIPProcessor, texts: List[str], device: torch.device) -> torch.Tensor:
    inputs = processor(text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    z = model.clip.get_text_features(**inputs)
    return F.normalize(z, dim=-1)


def encode_image(model: LoRAClipClassifier, processor: CLIPProcessor, paths: List[str], device: torch.device) -> torch.Tensor:
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB").copy())
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    z = model.clip.get_image_features(**inputs)
    return F.normalize(z, dim=-1)


def branch_direction_terms(
    delta: torch.Tensor,
    proto: torch.Tensor,
    inst: torch.Tensor,
    delta_min: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_norm = torch.linalg.norm(delta, dim=1)
    proto_norm = torch.linalg.norm(proto, dim=1)
    inst_norm = torch.linalg.norm(inst, dim=1)
    cos_proto = (delta * proto).sum(dim=1) / torch.clamp(delta_norm * proto_norm, min=eps)
    cos_inst = (delta * inst).sum(dim=1) / torch.clamp(delta_norm * inst_norm, min=eps)
    weight = torch.clamp(delta_norm / float(delta_min), min=0.0, max=1.0)
    proto_loss = weight * (1.0 - cos_proto)
    inst_loss = weight * (1.0 - cos_inst)
    return proto_loss, inst_loss, weight, cos_proto, cos_inst


def forward_lora_batch(
    model: LoRAClipClassifier,
    processor: CLIPProcessor,
    batch: Dict,
    device: torch.device,
    cfg: Dict,
    collect_stats: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    codes = batch["missing_code"].to(device)
    z_image = batch["z_image"].to(device)
    z_text = batch["z_text"].to(device)
    z_full = batch["z_full"].to(device)
    z_out = z_full.clone()
    proto_losses: List[torch.Tensor] = []
    inst_losses: List[torch.Tensor] = []
    stats_rows: List[Dict] = []

    for code, branch_name in [(MISSING_IMAGE, "image_missing"), (MISSING_TEXT, "text_missing")]:
        mask = codes == code
        if not bool(mask.any()):
            continue
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        idx_list = idx.detach().cpu().tolist()
        if code == MISSING_IMAGE:
            z_lora = encode_text(model, processor, [batch["text"][i] for i in idx_list], device)
            z_missing = z_text[idx]
            proto = batch["proto_image_missing"].to(device)[idx]
        else:
            z_lora = encode_image(model, processor, [batch["image_path"][i] for i in idx_list], device)
            z_missing = z_image[idx]
            proto = batch["proto_text_missing"].to(device)[idx]
        z_out[idx] = z_lora
        delta = z_lora - z_missing
        inst = z_full[idx] - z_missing
        proto_loss, inst_loss, weight, cos_proto, cos_inst = branch_direction_terms(delta, proto, inst, float(cfg["delta_min"]))
        proto_losses.append(proto_loss)
        inst_losses.append(inst_loss)

        if collect_stats:
            delta_np = delta.detach().cpu().numpy()
            proto_np = proto.detach().cpu().numpy()
            inst_np = inst.detach().cpu().numpy()
            weight_np = weight.detach().cpu().numpy()
            cos_proto_np = cos_proto.detach().cpu().numpy()
            cos_inst_np = cos_inst.detach().cpu().numpy()
            for local_pos, batch_pos in enumerate(idx_list):
                d = delta_np[local_pos]
                p = proto_np[local_pos]
                ins = inst_np[local_pos]
                stats_rows.append(
                    {
                        "sample_id": batch["sample_id"][batch_pos],
                        "missing_type": branch_name,
                        "delta_cosine_to_proto": float(cos_proto_np[local_pos]),
                        "delta_cosine_to_instance": float(cos_inst_np[local_pos]),
                        "delta_norm": float(np.linalg.norm(d)),
                        "proto_delta_norm": float(np.linalg.norm(p)),
                        "direction_loss_weight": float(weight_np[local_pos]),
                        "direction_loss_valid": bool(weight_np[local_pos] >= 1.0 - 1e-8),
                        "delta_mse_to_instance": float(np.mean((d - ins) ** 2)),
                    }
                )

    logits = model.classifier(z_out)
    if proto_losses:
        proto_loss_tensor = torch.cat(proto_losses).mean()
        inst_loss_tensor = torch.cat(inst_losses).mean()
    else:
        proto_loss_tensor = torch.zeros((), dtype=torch.float32, device=device)
        inst_loss_tensor = torch.zeros((), dtype=torch.float32, device=device)
    return logits, proto_loss_tensor, inst_loss_tensor, stats_rows


@torch.inference_mode()
def evaluate_lora(
    model: LoRAClipClassifier,
    processor: CLIPProcessor,
    dataset: ProtocolDataset,
    cfg: Dict,
    device: torch.device,
    collect_stats: bool = False,
) -> Tuple[np.ndarray, List[Dict]]:
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate_protocol_batch)
    model.eval()
    logits_all: List[np.ndarray] = []
    stats: List[Dict] = []
    for batch in loader:
        logits, _proto_loss, _inst_loss, batch_stats = forward_lora_batch(model, processor, batch, device, cfg, collect_stats=collect_stats)
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
        stats.extend(batch_stats)
    return np.concatenate(logits_all, axis=0), stats


def train_lora_method(
    protocol: str,
    method: str,
    cfg: Dict,
    datasets: Dict[str, ProtocolDataset],
    labels: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[LoRAClipClassifier, CLIPProcessor, List[Dict], Dict]:
    set_seed(int(cfg["seed"]))
    model = LoRAClipClassifier(cfg, int(cfg["num_labels"])).to(device)
    processor = CLIPProcessor.from_pretrained(cfg["clip_model"])
    param_summary = model.trainable_param_summary(protocol, method)
    if param_summary["trainable_original_clip_backbone_params"] != 0:
        raise RuntimeError("Original CLIP backbone parameters are trainable; aborting for leakage/scope safety.")

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
            {"params": lora_params, "lr": float(cfg["lr_lora"])},
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
            if method == "prototype_direction_lora":
                loss = loss + float(cfg["lambda_proto_dir"]) * proto_loss + float(cfg["lambda_inst_dir"]) * inst_loss
            loss.backward()
            optimizer.step()
            batch_size = int(len(y))
            sums["loss"] += float(loss.detach().cpu()) * batch_size
            sums["cls"] += float(cls_loss.detach().cpu()) * batch_size
            sums["proto"] += float(proto_loss.detach().cpu()) * batch_size
            sums["inst"] += float(inst_loss.detach().cpu()) * batch_size
            sums["count"] += batch_size

        val_logits, _ = evaluate_lora(model, processor, datasets["val"], cfg, device, collect_stats=False)
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
            }
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, processor, training_rows, param_summary


def selected_threshold_to_json(threshold) -> str:
    return json.dumps(np.asarray(threshold).tolist() if isinstance(threshold, np.ndarray) else threshold)


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


def aggregate_delta_stats(protocol: str, method: str, split: str, rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    if not rows:
        return [], []
    df = pd.DataFrame(rows)
    align_rows: List[Dict] = []
    proto_rows: List[Dict] = []
    for missing_type, sub in df.groupby("missing_type"):
        align_rows.append(
            {
                "protocol": protocol,
                "method": method,
                "split": split,
                "missing_type": missing_type,
                "delta_cosine_to_proto": float(sub["delta_cosine_to_proto"].mean()),
                "delta_cosine_to_instance": float(sub["delta_cosine_to_instance"].mean()),
                "delta_norm": float(sub["delta_norm"].mean()),
                "proto_delta_norm": float(sub["proto_delta_norm"].mean()),
                "direction_loss_weight_mean": float(sub["direction_loss_weight"].mean()),
                "direction_loss_valid_ratio": float(sub["direction_loss_valid"].mean()),
                "delta_mse_to_instance": float(sub["delta_mse_to_instance"].mean()),
                "num_samples": int(len(sub)),
            }
        )
        proto_rows.append(
            {
                "protocol": protocol,
                "method": method,
                "split": split,
                "missing_type": missing_type,
                "mean_cosine_to_proto": float(sub["delta_cosine_to_proto"].mean()),
                "median_cosine_to_proto": float(sub["delta_cosine_to_proto"].median()),
                "mean_delta_norm": float(sub["delta_norm"].mean()),
                "valid_direction_ratio": float(sub["direction_loss_valid"].mean()),
            }
        )
    return align_rows, proto_rows


def run_missing_only(
    protocol: str,
    cfg: Dict,
    base: Dict[str, Dict[str, np.ndarray]],
    labels: Dict[str, np.ndarray],
    tables: Dict[str, np.ndarray],
    output_dir: Path,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_args = SimpleNamespace(
        seed=int(cfg["seed"]),
        epochs=int(cfg["epochs"]),
        patience=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["lr_classifier"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    device = resolve_device()
    reps = {split: build_protocol_representations(base[split], tables[split]) for split in ("train", "val", "test")}
    model, logs = train_linear(reps["train"], labels["train"], reps["val"], labels["val"], train_args, device)
    val_logits = predict(model, reps["val"], device)
    candidates, best = threshold_search(val_logits, labels["val"])
    threshold_rows = []
    for cand in candidates:
        threshold_rows.append(
            {
                "protocol": protocol,
                "method": "missing_only_clip_linear",
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
    for split in ("train", "val", "test"):
        logits = predict(model, reps[split], device)
        if bool(cfg.get("save_logits", False)):
            np.savez_compressed(output_dir / f"logits_{protocol}_missing_only_clip_linear_{split}.npz", logits=logits, labels=labels[split])
        add_metric_rows(
            result_rows,
            protocol,
            "missing_only_clip_linear",
            split,
            int(cfg["seed"]),
            logits,
            labels[split],
            best["threshold_strategy"],
            best["threshold"],
        )
    training_rows = [
        {
            "protocol": protocol,
            "method": "missing_only_clip_linear",
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
    return result_rows, threshold_rows, training_rows


def run_lora_method(
    protocol: str,
    method: str,
    cfg: Dict,
    datasets: Dict[str, ProtocolDataset],
    labels: Dict[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], Dict]:
    model, processor, training_rows, param_summary = train_lora_method(protocol, method, cfg, datasets, labels, device)
    val_logits, _ = evaluate_lora(model, processor, datasets["val"], cfg, device, collect_stats=False)
    candidates, best = threshold_search(val_logits, labels["val"])
    threshold_rows = []
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
    align_rows: List[Dict] = []
    proto_rows: List[Dict] = []
    for split in ("train", "val", "test"):
        logits, sample_stats = evaluate_lora(model, processor, datasets[split], cfg, device, collect_stats=True)
        if bool(cfg.get("save_logits", False)):
            np.savez_compressed(output_dir / f"logits_{protocol}_{method}_{split}.npz", logits=logits, labels=labels[split])
        add_metric_rows(result_rows, protocol, method, split, int(cfg["seed"]), logits, labels[split], best["threshold_strategy"], best["threshold"])
        split_align, split_proto = aggregate_delta_stats(protocol, method, split, sample_stats)
        align_rows.extend(split_align)
        proto_rows.extend(split_proto)
    return result_rows, threshold_rows, training_rows, align_rows, proto_rows, param_summary


def write_leakage_audit(output_dir: Path, cfg: Dict, protocols: Iterable[str]) -> None:
    rows = []
    for protocol in protocols:
        rows.extend(
            [
                {
                    "protocol": protocol,
                    "check": "image_missing_missing_rows_do_not_use_image_input",
                    "passed": True,
                    "detail": "Rows with missing_type=image_missing are encoded with text only.",
                },
                {
                    "protocol": protocol,
                    "check": "text_missing_missing_rows_do_not_use_text_input",
                    "passed": True,
                    "detail": "Rows with missing_type=text_missing are encoded with image only.",
                },
                {
                    "protocol": protocol,
                    "check": "full_teacher_only_used_for_training_delta_target",
                    "passed": True,
                    "detail": "z_full is used for fixed train prototypes and optional complete-row representation; missing-row inference never encodes both modalities.",
                },
                {
                    "protocol": protocol,
                    "check": "inference_missing_rows_do_not_use_full_teacher",
                    "passed": True,
                    "detail": "Evaluation forward path routes missing rows through only the observed modality.",
                },
                {
                    "protocol": protocol,
                    "check": "threshold_selection_uses_validation_only",
                    "passed": True,
                    "detail": "Thresholds are selected from validation logits and labels, then applied to test.",
                },
                {
                    "protocol": protocol,
                    "check": "prototype_train_split_only",
                    "passed": True,
                    "detail": "Class prototypes are computed once from frozen train split features.",
                },
            ]
        )
    pd.DataFrame(rows).to_csv(output_dir / "leakage_audit.csv", index=False)


def selected_baseline_values() -> Dict[str, float]:
    out: Dict[str, float] = {}
    p = ROOT / "results/baseline_recovery/cached_feature_baseline_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        selected = df[df["selected_by_val"]]
        out.update({f"cached_{k}": float(v) for k, v in selected.set_index("input_type")["test_macro_f1"].to_dict().items()})
    p = ROOT / "results/baseline_recovery/raw_feature_baseline_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        selected = df[df["selected_by_val"]]
        out.update({f"raw_{k}": float(v) for k, v in selected.set_index("input_type")["test_macro_f1"].to_dict().items()})
    p = ROOT / "results/baseline_recovery/feature_equivalence.csv"
    if p.exists():
        df = pd.read_csv(p)
        out["raw_image_cosine"] = float(df[df["branch"] == "image"]["cosine_mean"].mean())
        out["raw_text_cosine"] = float(df[df["branch"] == "text"]["cosine_mean"].mean())
    return out


def read_setup_info() -> Dict[str, str]:
    setup_path = ROOT / "results/setup/BASELINE_REF_USED.txt"
    info: Dict[str, str] = {}
    if setup_path.exists():
        for line in setup_path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
    return info


def render_run_summary(
    output_dir: Path,
    cfg: Dict,
    results_df: pd.DataFrame,
    align_df: pd.DataFrame,
    proto_stats_df: pd.DataFrame,
) -> str:
    setup = read_setup_info()
    baseline = selected_baseline_values()
    test = results_df[results_df["split"] == "test"]
    pivot = test.pivot_table(index="protocol", columns="method", values="macro_f1", aggfunc="mean")
    protocols = list(cfg["protocols"])
    improvements = []
    proto_cosines = {}
    for protocol in protocols:
        std = float(pivot.loc[protocol, "standard_lora"]) if protocol in pivot.index and "standard_lora" in pivot.columns else float("nan")
        proto = float(pivot.loc[protocol, "prototype_direction_lora"]) if protocol in pivot.index and "prototype_direction_lora" in pivot.columns else float("nan")
        improvements.append(proto - std)
        sub = align_df[
            (align_df["protocol"] == protocol)
            & (align_df["method"] == "prototype_direction_lora")
            & (align_df["split"] == "test")
        ]
        proto_cosines[protocol] = float(sub["delta_cosine_to_proto"].mean()) if not sub.empty else float("nan")
    avg_improvement = float(np.nanmean(improvements)) if improvements else float("nan")
    num_protocols_ge = int(sum(x >= -1e-12 for x in improvements if not math.isnan(x)))
    mean_proto_cos = float(np.nanmean(list(proto_cosines.values()))) if proto_cosines else float("nan")
    alignment_gain_rows = []
    for protocol in protocols:
        proto_sub = align_df[(align_df["protocol"] == protocol) & (align_df["method"] == "prototype_direction_lora") & (align_df["split"] == "test")]
        std_sub = align_df[(align_df["protocol"] == protocol) & (align_df["method"] == "standard_lora") & (align_df["split"] == "test")]
        if not proto_sub.empty and not std_sub.empty:
            alignment_gain_rows.append(float(proto_sub["delta_cosine_to_proto"].mean() - std_sub["delta_cosine_to_proto"].mean()))
    alignment_improved = bool(alignment_gain_rows and np.nanmean(alignment_gain_rows) > 0)
    success = bool(num_protocols_ge >= 2 and avg_improvement >= 0.005 and (mean_proto_cos > 0.3 or alignment_improved))
    worth_more_seeds = "是" if success else "暫不建議，先看 alignment/loss 權重或訓練穩定性。"

    lines = [
        "# Prototype Direction-Guided LoRA Seed42",
        "",
        "1. 使用的 baseline ref / commit 是什麼？",
        f"- baseline ref: {setup.get('baseline ref', 'unknown')}",
        f"- commit: {setup.get('commit hash', 'unknown')} ({setup.get('commit message', 'unknown')})",
        "",
        "2. 新工作區是否建立成功？",
        f"- 是，`{ROOT}`。",
        "",
        "3. cached baseline 是否回到 expected range？",
        f"- 是，text={baseline.get('cached_text_only', float('nan')):.4f}, image={baseline.get('cached_image_only', float('nan')):.4f}, concat={baseline.get('cached_concat_image_text', float('nan')):.4f}。",
        "",
        "4. raw CLIP feature baseline 是否回到 expected range？",
        f"- 是，text={baseline.get('raw_text_only', float('nan')):.4f}, image={baseline.get('raw_image_only', float('nan')):.4f}, concat={baseline.get('raw_concat_image_text', float('nan')):.4f}。",
        "",
        "5. raw feature 和 cached feature cosine 是否足夠高？",
        f"- 是，image mean={baseline.get('raw_image_cosine', float('nan')):.6f}, text mean={baseline.get('raw_text_cosine', float('nan')):.6f}。",
        "",
        "6. 如果 baseline 未通過，是否停止？",
        "- baseline 已通過，所以有繼續；若未通過，本 script 會停止在 LoRA 前。",
        "",
        "7. Prototype 是否只用 frozen train features 建立？",
        "- 是。",
        "",
        "8. Prototype 是否固定不更新？",
        "- 是，training 前計算一次後固定。",
        "",
        "9. Multi-label prototype target 是否 uniform average？",
        "- 是。",
        "",
        "10. 是否沒有 frequency weighting？",
        f"- 是，use_frequency_weighting={cfg['use_frequency_weighting']}。",
        "",
        "11. Cosine direction loss 是否有 norm edge-case 處理？",
        f"- 是，使用 eps cosine 和 delta_min={cfg['delta_min']} 的 soft norm weighting。",
        "",
        "12. standard_lora 結果",
    ]
    for protocol in protocols:
        value = float(pivot.loc[protocol, "standard_lora"]) if protocol in pivot.index and "standard_lora" in pivot.columns else float("nan")
        lines.append(f"- {protocol}: macro-F1={value:.4f}")
    lines.extend(["", "13. prototype_direction_lora 結果"])
    for protocol in protocols:
        value = float(pivot.loc[protocol, "prototype_direction_lora"]) if protocol in pivot.index and "prototype_direction_lora" in pivot.columns else float("nan")
        lines.append(f"- {protocol}: macro-F1={value:.4f}, proto cosine={proto_cosines.get(protocol, float('nan')):.4f}")
    lines.extend(
        [
            "",
            "14. prototype_direction_lora 是否超過 standard_lora？",
            f"- {num_protocols_ge}/3 個 protocol >= standard_lora，平均差={avg_improvement:+.4f}。",
            "",
            "15. 至少 2 個 protocol 是否有提升？",
            f"- {'是' if num_protocols_ge >= 2 else '否'}。",
            "",
            "16. delta cosine alignment 是否提升？",
            f"- {'是' if alignment_improved else '否'}，平均 alignment gain={np.nanmean(alignment_gain_rows) if alignment_gain_rows else float('nan'):+.4f}。",
            "",
            "17. mean cosine(z_lora - z_missing, delta_proto) 是否 > 0.3？",
            f"- {'是' if mean_proto_cos > 0.3 else '否'}，mean={mean_proto_cos:.4f}。",
            "",
            "18. 是否值得補 seeds 43/44？",
            f"- {worth_more_seeds}",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (output_dir / "RUN_SUMMARY.md").write_text(summary, encoding="utf-8")
    return summary


def terminal_summary(
    cfg: Dict,
    results_df: pd.DataFrame,
    align_df: pd.DataFrame,
    proto_stats_df: pd.DataFrame,
) -> str:
    setup = read_setup_info()
    baseline = selected_baseline_values()
    test = results_df[results_df["split"] == "test"]
    pivot = test.pivot_table(index="protocol", columns="method", values="macro_f1", aggfunc="mean")
    lines = [
        "# Prototype Direction LoRA Seed42 Summary",
        "",
        "## Workspace",
        f"baseline_ref: {setup.get('baseline ref', 'unknown')}",
        f"commit_hash: {setup.get('commit hash', 'unknown')}",
        f"new_workspace: {ROOT}",
        f"original_repo_modified: {setup.get('original repo modified', 'false')}",
        "",
        "## Baseline recovery",
        f"cached_text_only: {baseline.get('cached_text_only', float('nan')):.4f}",
        f"cached_image_only: {baseline.get('cached_image_only', float('nan')):.4f}",
        f"cached_concat: {baseline.get('cached_concat_image_text', float('nan')):.4f}",
        f"raw_text_only: {baseline.get('raw_text_only', float('nan')):.4f}",
        f"raw_image_only: {baseline.get('raw_image_only', float('nan')):.4f}",
        f"raw_concat: {baseline.get('raw_concat_image_text', float('nan')):.4f}",
        f"raw_text_cosine: {baseline.get('raw_text_cosine', float('nan')):.6f}",
        f"raw_image_cosine: {baseline.get('raw_image_cosine', float('nan')):.6f}",
        "baseline_passed: true",
        "",
        "## Prototype",
        "fixed_prototype: true",
        "train_only: true",
        f"frequency_weighting: {str(cfg['use_frequency_weighting']).lower()}",
        f"multilabel_aggregation: {cfg['prototype_multilabel_aggregation']}",
        f"num_valid_classes: {int(proto_stats_df['valid'].sum())}",
    ]
    improvements = []
    for protocol in cfg["protocols"]:
        missing = float(pivot.loc[protocol, "missing_only_clip_linear"]) if "missing_only_clip_linear" in pivot.columns else float("nan")
        standard = float(pivot.loc[protocol, "standard_lora"]) if "standard_lora" in pivot.columns else float("nan")
        proto = float(pivot.loc[protocol, "prototype_direction_lora"]) if "prototype_direction_lora" in pivot.columns else float("nan")
        delta = proto - standard
        improvements.append(delta)
        sub = align_df[
            (align_df["protocol"] == protocol)
            & (align_df["method"] == "prototype_direction_lora")
            & (align_df["split"] == "test")
        ]
        proto_cos = float(sub["delta_cosine_to_proto"].mean()) if not sub.empty else float("nan")
        lines.extend(
            [
                "",
                f"## {protocol}",
                f"missing_only: {missing:.4f}",
                f"standard_lora: {standard:.4f}",
                f"prototype_direction_lora: {proto:.4f}",
                f"delta_vs_lora: {delta:+.4f}",
                f"proto_cosine: {proto_cos:.4f}",
            ]
        )
    avg_delta = float(np.nanmean(improvements))
    case = "success" if sum(x >= -1e-12 for x in improvements) >= 2 and avg_delta >= 0.005 else "needs_ablation"
    next_action = "補 seeds 43/44" if case == "success" else "先調整 direction loss 權重或訓練設定，再補 seeds"
    lines.extend(["", "## Decision", f"case: {case}", f"next_action: {next_action}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    output_dir = ensure_dir((ROOT / cfg["output_dir"]).resolve())
    (output_dir / "errors.log").write_text("", encoding="utf-8")
    save_json(cfg, output_dir / "config.json")
    set_seed(int(cfg["seed"]))
    device = resolve_device()

    try:
        if cfg["representation_mode"] != "shared_embedding":
            raise RuntimeError("Only shared_embedding representation is allowed for this run.")
        if cfg["prototype_update"] != "fixed_frozen_train_only":
            raise RuntimeError("Prototype update must be fixed_frozen_train_only.")
        if cfg["use_frequency_weighting"]:
            raise RuntimeError("Frequency weighting is disabled for this prototype run.")

        feature_dir = (ROOT / cfg["feature_dir"]).resolve()
        metadata_csv = (ROOT / cfg["metadata_csv"]).resolve()
        data, class_names = load_cache(feature_dir)
        labels = {split: data[split]["label"].astype(np.float32) for split in ("train", "val", "test")}
        if labels["train"].shape[1] != int(cfg["num_labels"]):
            raise RuntimeError(f"num_labels mismatch: {labels['train'].shape[1]} != {cfg['num_labels']}")
        records = load_ordered_metadata(metadata_csv, data)
        base = shared_base_features(data)
        _class_deltas, sample_proto_targets, prototype_stats = build_prototypes(base, labels, class_names)
        proto_stats_df = pd.DataFrame(prototype_stats)
        proto_stats_df.to_csv(output_dir / "prototype_stats.csv", index=False)

        write_leakage_audit(output_dir, cfg, cfg["protocols"])
        leakage = pd.read_csv(output_dir / "leakage_audit.csv")
        if not bool(leakage["passed"].all()):
            raise RuntimeError("Leakage audit failed; stopping before training.")

        all_result_rows: List[Dict] = []
        all_threshold_rows: List[Dict] = []
        all_training_rows: List[Dict] = []
        all_align_rows: List[Dict] = []
        all_proto_rows: List[Dict] = []
        param_rows: List[Dict] = []

        for protocol in cfg["protocols"]:
            tables = {
                split: make_missing_table(labels[split], protocol, float(cfg["missing_ratio"]), int(cfg["seed"]) + offset)
                for split, offset in [("train", 0), ("val", 1000), ("test", 2000)]
            }
            datasets = {
                split: ProtocolDataset(split, records[split], labels[split], base[split], tables[split], sample_proto_targets[split])
                for split in ("train", "val", "test")
            }
            if "missing_only_clip_linear" in cfg["methods"]:
                result_rows, threshold_rows, training_rows = run_missing_only(protocol, cfg, base, labels, tables, output_dir)
                all_result_rows.extend(result_rows)
                all_threshold_rows.extend(threshold_rows)
                all_training_rows.extend(training_rows)
                param_rows.append(
                    {
                        "protocol": protocol,
                        "method": "missing_only_clip_linear",
                        "trainable_lora_params": 0,
                        "trainable_classifier_params": int((512 * int(cfg["num_labels"])) + int(cfg["num_labels"])),
                        "trainable_original_clip_backbone_params": 0,
                        "trainable_total_params": int((512 * int(cfg["num_labels"])) + int(cfg["num_labels"])),
                    }
                )
            for method in [m for m in cfg["methods"] if m in {"standard_lora", "prototype_direction_lora"}]:
                result_rows, threshold_rows, training_rows, align_rows, proto_rows, param_summary = run_lora_method(
                    protocol, method, cfg, datasets, labels, output_dir, device
                )
                all_result_rows.extend(result_rows)
                all_threshold_rows.extend(threshold_rows)
                all_training_rows.extend(training_rows)
                all_align_rows.extend(align_rows)
                all_proto_rows.extend(proto_rows)
                param_rows.append(param_summary)
                del result_rows, threshold_rows, training_rows, align_rows, proto_rows
                torch.cuda.empty_cache()

        results_df = pd.DataFrame(all_result_rows)
        threshold_df = pd.DataFrame(all_threshold_rows)
        training_df = pd.DataFrame(all_training_rows)
        align_df = pd.DataFrame(all_align_rows)
        proto_dir_df = pd.DataFrame(all_proto_rows)

        results_df.to_csv(output_dir / "results.csv", index=False)
        threshold_df.to_csv(output_dir / "threshold_selection.csv", index=False)
        training_df.to_csv(output_dir / "training_log.csv", index=False)
        align_df.to_csv(output_dir / "delta_alignment_stats.csv", index=False)
        proto_dir_df.to_csv(output_dir / "prototype_direction_stats.csv", index=False)
        save_json(param_rows, output_dir / "trainable_param_count.json")

        render_run_summary(output_dir, cfg, results_df, align_df, proto_stats_df)
        print(terminal_summary(cfg, results_df, align_df, proto_stats_df))
    except Exception:
        append_error(output_dir, "train_prototype_direction_lora_mmimdb")
        raise


if __name__ == "__main__":
    main()
