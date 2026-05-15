import argparse
import json
import random
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover cached MM-IMDb feature baselines.")
    parser.add_argument("--feature-dir", default="cache/mmimdb_multilabel_clean")
    parser.add_argument("--output-dir", default="results/baseline_recovery")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_error(output_dir: Path, title: str) -> None:
    with (output_dir / "errors.log").open("a", encoding="utf-8") as f:
        f.write(f"\n[{title}]\n")
        f.write(traceback.format_exc())
        f.write("\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def apply_thresholds(prob: np.ndarray, thresholds) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim == 0:
        return (prob >= float(thresholds)).astype(np.float32)
    return (prob >= thresholds.reshape(1, -1)).astype(np.float32)


def mean_bce_loss_np(logits: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    prob = sigmoid_np(logits)
    loss = -(y * np.log(np.clip(prob, eps, 1.0 - eps)) + (1.0 - y) * np.log(np.clip(1.0 - prob, eps, 1.0 - eps)))
    return float(loss.mean())


def multilabel_metrics(logits: np.ndarray, y: np.ndarray, thresholds) -> Dict[str, float]:
    pred = apply_thresholds(sigmoid_np(logits), thresholds)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "sample_f1": float(f1_score(y, pred, average="samples", zero_division=0)),
        "mean_predicted_positive_labels": float(pred.sum(axis=1).mean()),
        "mean_true_positive_labels": float(y.sum(axis=1).mean()),
    }


def threshold_grid() -> List[float]:
    return [round(i / 100, 2) for i in range(5, 100, 5)]


def threshold_search(logits: np.ndarray, y: np.ndarray) -> Tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    best = None
    prob = sigmoid_np(logits)
    for t in threshold_grid():
        metrics = multilabel_metrics(logits, y, t)
        row = {
            "threshold_strategy": "global",
            "threshold": float(t),
            "val_macro_f1": metrics["macro_f1"],
            "val_micro_f1": metrics["micro_f1"],
            "val_sample_f1": metrics["sample_f1"],
            "val_bce_loss": mean_bce_loss_np(logits, y),
        }
        rows.append(row)
        if best is None or row["val_macro_f1"] > best["val_macro_f1"]:
            best = row

    thresholds = np.zeros(y.shape[1], dtype=np.float32)
    for j in range(y.shape[1]):
        best_t = 0.5
        best_f1 = -1.0
        for t in threshold_grid():
            pred_j = (prob[:, j] >= float(t)).astype(np.float32)
            score = f1_score(y[:, j], pred_j, average="binary", zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = float(t)
        thresholds[j] = best_t
    metrics = multilabel_metrics(logits, y, thresholds)
    row = {
        "threshold_strategy": "per_class",
        "threshold": thresholds,
        "val_macro_f1": metrics["macro_f1"],
        "val_micro_f1": metrics["micro_f1"],
        "val_sample_f1": metrics["sample_f1"],
        "val_bce_loss": mean_bce_loss_np(logits, y),
    }
    rows.append(row)
    if best is None or row["val_macro_f1"] > best["val_macro_f1"]:
        best = row
    assert best is not None
    return rows, best


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.net = nn.Linear(input_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def positive_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    weights = np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.inference_mode()
def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(x), 512):
        batch = torch.tensor(x[start : start + 512], dtype=torch.float32, device=device)
        outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def train_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, List[Dict]]:
    set_seed(args.seed)
    model = LinearHead(x_train.shape[1], y_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight(y_train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_state = None
    best_score = -1.0
    stale = 0
    log_rows: List[Dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_x)
            total_count += len(batch_x)
        val_logits = predict(model, x_val, device)
        _, best = threshold_search(val_logits, y_val)
        score = float(best["val_macro_f1"])
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_count, 1),
                "val_macro_f1": score,
                "best_threshold_strategy": best["threshold_strategy"],
            }
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, log_rows


def load_cache(feature_dir: Path) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        data[split] = {
            "image": np.load(feature_dir / f"{split}_image.npy", allow_pickle=True).astype(np.float32),
            "text": np.load(feature_dir / f"{split}_text.npy", allow_pickle=True).astype(np.float32),
            "label": np.load(feature_dir / f"{split}_label.npy", allow_pickle=True).astype(np.float32),
            "sample_ids": np.load(feature_dir / f"{split}_sample_ids.npy", allow_pickle=True).astype(str),
        }
    meta_path = feature_dir / "feature_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    class_names = meta.get("class_names") or [f"class_{i}" for i in range(data["train"]["label"].shape[1])]
    return data, class_names


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    feature_dir = (root / args.feature_dir).resolve()
    output_dir = ensure_dir((root / args.output_dir).resolve())
    (output_dir / "errors.log").write_text("", encoding="utf-8")
    device = resolve_device(args.device)
    set_seed(args.seed)

    try:
        data, _class_names = load_cache(feature_dir)
        feature_sets = {
            "text_only": {split: data[split]["text"] for split in ("train", "val", "test")},
            "image_only": {split: data[split]["image"] for split in ("train", "val", "test")},
            "concat_image_text": {
                split: np.concatenate([data[split]["image"], data[split]["text"]], axis=1).astype(np.float32)
                for split in ("train", "val", "test")
            },
        }

        result_rows: List[Dict] = []
        threshold_rows: List[Dict] = []
        training_rows: List[Dict] = []
        for input_type, features in feature_sets.items():
            model, logs = train_linear(
                features["train"],
                data["train"]["label"],
                features["val"],
                data["val"]["label"],
                args,
                device,
            )
            for row in logs:
                row = dict(row)
                row["input_type"] = input_type
                row["seed"] = args.seed
                training_rows.append(row)

            val_logits = predict(model, features["val"], device)
            test_logits = predict(model, features["test"], device)
            candidates, best = threshold_search(val_logits, data["val"]["label"])
            best_threshold_strategy = best["threshold_strategy"]
            for cand in candidates:
                test_metrics = multilabel_metrics(test_logits, data["test"]["label"], cand["threshold"])
                threshold_value = cand["threshold"]
                threshold_rows.append(
                    {
                        "input_type": input_type,
                        "seed": args.seed,
                        "threshold_strategy": cand["threshold_strategy"],
                        "threshold": json.dumps(np.asarray(threshold_value).tolist() if isinstance(threshold_value, np.ndarray) else threshold_value),
                        "val_macro_f1": cand["val_macro_f1"],
                        "val_micro_f1": cand["val_micro_f1"],
                        "val_sample_f1": cand["val_sample_f1"],
                        "val_bce_loss": cand["val_bce_loss"],
                    }
                )
                result_rows.append(
                    {
                        "input_type": input_type,
                        "seed": args.seed,
                        "threshold_strategy": cand["threshold_strategy"],
                        "selected_by_val": cand["threshold_strategy"] == best_threshold_strategy,
                        "val_macro_f1": cand["val_macro_f1"],
                        "test_macro_f1": test_metrics["macro_f1"],
                        "test_micro_f1": test_metrics["micro_f1"],
                        "test_sample_f1": test_metrics["sample_f1"],
                        "test_bce_loss": mean_bce_loss_np(test_logits, data["test"]["label"]),
                        "mean_predicted_positive_labels": test_metrics["mean_predicted_positive_labels"],
                        "mean_true_positive_labels": test_metrics["mean_true_positive_labels"],
                        "epochs_ran": len(logs),
                        "device": str(device),
                    }
                )

        pd.DataFrame(result_rows).to_csv(output_dir / "cached_feature_baseline_results.csv", index=False)
        pd.DataFrame(threshold_rows).to_csv(output_dir / "cached_feature_threshold_selection.csv", index=False)
        pd.DataFrame(training_rows).to_csv(output_dir / "cached_feature_training_log.csv", index=False)
    except Exception:
        append_error(output_dir, "recover_cached_feature_baseline")
        raise


if __name__ == "__main__":
    main()
