import argparse
import json
import random
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean MM-IMDb multi-label baselines.")
    parser.add_argument("--feature-dir", default="cache/mmimdb_multilabel_clean")
    parser.add_argument("--output-dir", default="results/mmimdb_multilabel_clean_baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--threshold-strategies", nargs="+", default=["global", "per_class"], choices=["global", "per_class"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def apply_thresholds_np(prob: np.ndarray, thresholds) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim == 0:
        return (prob >= float(thresholds)).astype(np.float32)
    return (prob >= thresholds.reshape(1, -1)).astype(np.float32)


def mean_bce_loss_np(logits: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    prob = sigmoid_np(logits)
    loss = -(y * np.log(np.clip(prob, eps, 1.0 - eps)) + (1.0 - y) * np.log(np.clip(1.0 - prob, eps, 1.0 - eps)))
    return float(loss.mean())


def multilabel_metrics_np(logits: np.ndarray, y: np.ndarray, thresholds) -> Dict[str, float]:
    prob = sigmoid_np(logits)
    pred = apply_thresholds_np(prob, thresholds)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "sample_f1": float(f1_score(y, pred, average="samples", zero_division=0)),
        "mean_predicted_positive_labels": float(pred.sum(axis=1).mean()),
        "mean_true_positive_labels": float(y.sum(axis=1).mean()),
        "pred": pred,
    }


def select_threshold_candidates() -> List[float]:
    return [round(i / 100, 2) for i in range(5, 100, 5)]


def threshold_search(logits: np.ndarray, y: np.ndarray, strategies: List[str]) -> Tuple[List[Dict], Dict]:
    grid = select_threshold_candidates()
    prob = sigmoid_np(logits)
    rows = []
    best = None
    if "global" in strategies:
        for t in grid:
            metrics = multilabel_metrics_np(logits, y, t)
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
    if "per_class" in strategies:
        thresholds = np.zeros(y.shape[1], dtype=np.float32)
        for j in range(y.shape[1]):
            best_t = 0.5
            best_f1 = -1.0
            for t in grid:
                pred_j = (prob[:, j] >= float(t)).astype(np.float32)
                score = f1_score(y[:, j], pred_j, average="binary", zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = float(t)
            thresholds[j] = best_t
        metrics = multilabel_metrics_np(logits, y, thresholds)
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


def load_clean_cache(feature_dir: Path):
    data = {}
    for split in ("train", "val", "test"):
        data[split] = {
            "image": np.load(feature_dir / f"{split}_image.npy", allow_pickle=True).astype(np.float32),
            "text": np.load(feature_dir / f"{split}_text.npy", allow_pickle=True).astype(np.float32),
            "label": np.load(feature_dir / f"{split}_label.npy", allow_pickle=True).astype(np.float32),
            "sample_ids": np.load(feature_dir / f"{split}_sample_ids.npy", allow_pickle=True).astype(str),
        }
    meta_path = feature_dir / "feature_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    class_names = meta.get("class_names")
    if not class_names:
        class_names = [f"class_{i}" for i in range(data["train"]["label"].shape[1])]
    return data, class_names, meta


def decision_scores_from_ovr(clf, x: np.ndarray) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(x)
    elif hasattr(clf, "predict_proba"):
        prob = np.clip(clf.predict_proba(x), 1e-6, 1 - 1e-6)
        scores = np.log(prob / (1.0 - prob))
    else:
        raise ValueError("classifier has neither decision_function nor predict_proba")
    return np.asarray(scores, dtype=np.float32)


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def positive_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    weights = np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.inference_mode()
def predict_torch(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(x), 512):
        batch = torch.tensor(x[start : start + 512], dtype=torch.float32, device=device)
        logits = model(batch)
        outputs.append(logits.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> nn.Module:
    set_seed(seed)
    model.to(device)
    pos_weight = positive_weight(y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state = None
    best_score = -1.0
    stale = 0
    for _epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        val_logits = predict_torch(model, x_val, device)
        _, best = threshold_search(val_logits, y_val, args.threshold_strategies)
        score = float(best["val_macro_f1"])
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
    return model


def per_label_rows(model_type: str, input_type: str, seed: int, threshold_strategy: str, logits: np.ndarray, y: np.ndarray, thresholds, class_names: List[str]) -> List[Dict]:
    prob = sigmoid_np(logits)
    pred = apply_thresholds_np(prob, thresholds)
    rows = []
    for j, name in enumerate(class_names):
        rows.append(
            {
                "model_type": model_type,
                "input_type": input_type,
                "seed": seed,
                "threshold_strategy": threshold_strategy,
                "label_id": j,
                "label_name": name,
                "f1": float(f1_score(y[:, j], pred[:, j], average="binary", zero_division=0)),
                "support": int(y[:, j].sum()),
                "predicted_positive": int(pred[:, j].sum()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    feature_dir = (repo_root / args.feature_dir).resolve()
    output_dir = ensure_dir((repo_root / args.output_dir).resolve())
    write_text(output_dir / "errors.log", "")
    device = resolve_device(args.device)

    try:
        data, class_names, meta = load_clean_cache(feature_dir)
        features = {
            "text_only": {
                split: data[split]["text"].astype(np.float32) for split in ("train", "val", "test")
            },
            "image_only": {
                split: data[split]["image"].astype(np.float32) for split in ("train", "val", "test")
            },
            "concat_image_text": {
                split: np.concatenate([data[split]["image"], data[split]["text"]], axis=1).astype(np.float32)
                for split in ("train", "val", "test")
            },
        }

        baseline_rows: List[Dict] = []
        threshold_rows: List[Dict] = []
        per_label_all: List[Dict] = []
        prediction_dist_rows: List[Dict] = []

        for seed in args.seeds:
            set_seed(seed)
            for input_type, split_x in features.items():
                x_train, x_val, x_test = split_x["train"], split_x["val"], split_x["test"]
                y_train, y_val, y_test = data["train"]["label"], data["val"]["label"], data["test"]["label"]

                # sklearn logistic
                logistic = OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    )
                )
                logistic.fit(x_train, y_train)
                val_logits = decision_scores_from_ovr(logistic, x_val)
                test_logits = decision_scores_from_ovr(logistic, x_test)
                candidates, best = threshold_search(val_logits, y_val, args.threshold_strategies)
                for cand in candidates:
                    test_metrics = multilabel_metrics_np(test_logits, y_test, cand["threshold"])
                    baseline_rows.append(
                        {
                            "model_type": "logistic_ovr",
                            "input_type": input_type,
                            "seed": seed,
                            "threshold_strategy": cand["threshold_strategy"],
                            "val_macro_f1": cand["val_macro_f1"],
                            "test_macro_f1": test_metrics["macro_f1"],
                            "test_micro_f1": test_metrics["micro_f1"],
                            "test_sample_f1": test_metrics["sample_f1"],
                            "mean_predicted_positive_labels": test_metrics["mean_predicted_positive_labels"],
                            "mean_true_positive_labels": test_metrics["mean_true_positive_labels"],
                            "bce_loss": mean_bce_loss_np(test_logits, y_test),
                        }
                    )
                    threshold_rows.append(
                        {
                            "model_type": "logistic_ovr",
                            "input_type": input_type,
                            "seed": seed,
                            "threshold_strategy": cand["threshold_strategy"],
                            "threshold": json.dumps(np.asarray(cand["threshold"]).tolist() if isinstance(cand["threshold"], np.ndarray) else cand["threshold"]),
                            "val_macro_f1": cand["val_macro_f1"],
                            "val_micro_f1": cand["val_micro_f1"],
                            "val_sample_f1": cand["val_sample_f1"],
                            "val_bce_loss": cand["val_bce_loss"],
                        }
                    )
                    per_label_all.extend(per_label_rows("logistic_ovr", input_type, seed, cand["threshold_strategy"], test_logits, y_test, cand["threshold"], class_names))
                    prediction_dist_rows.append(
                        {
                            "model_type": "logistic_ovr",
                            "input_type": input_type,
                            "seed": seed,
                            "threshold_strategy": cand["threshold_strategy"],
                            "mean_predicted_positive_labels": test_metrics["mean_predicted_positive_labels"],
                            "mean_true_positive_labels": test_metrics["mean_true_positive_labels"],
                        }
                    )

                # torch linear
                linear = LinearHead(x_train.shape[1], y_train.shape[1])
                linear = train_torch_model(linear, x_train, y_train, x_val, y_val, args, seed, device)
                val_logits = predict_torch(linear, x_val, device)
                test_logits = predict_torch(linear, x_test, device)
                candidates, _best = threshold_search(val_logits, y_val, args.threshold_strategies)
                for cand in candidates:
                    test_metrics = multilabel_metrics_np(test_logits, y_test, cand["threshold"])
                    baseline_rows.append(
                        {
                            "model_type": "torch_linear",
                            "input_type": input_type,
                            "seed": seed,
                            "threshold_strategy": cand["threshold_strategy"],
                            "val_macro_f1": cand["val_macro_f1"],
                            "test_macro_f1": test_metrics["macro_f1"],
                            "test_micro_f1": test_metrics["micro_f1"],
                            "test_sample_f1": test_metrics["sample_f1"],
                            "mean_predicted_positive_labels": test_metrics["mean_predicted_positive_labels"],
                            "mean_true_positive_labels": test_metrics["mean_true_positive_labels"],
                            "bce_loss": mean_bce_loss_np(test_logits, y_test),
                        }
                    )
                    threshold_rows.append(
                        {
                            "model_type": "torch_linear",
                            "input_type": input_type,
                            "seed": seed,
                            "threshold_strategy": cand["threshold_strategy"],
                            "threshold": json.dumps(np.asarray(cand["threshold"]).tolist() if isinstance(cand["threshold"], np.ndarray) else cand["threshold"]),
                            "val_macro_f1": cand["val_macro_f1"],
                            "val_micro_f1": cand["val_micro_f1"],
                            "val_sample_f1": cand["val_sample_f1"],
                            "val_bce_loss": cand["val_bce_loss"],
                        }
                    )
                    per_label_all.extend(per_label_rows("torch_linear", input_type, seed, cand["threshold_strategy"], test_logits, y_test, cand["threshold"], class_names))
                    prediction_dist_rows.append(
                        {
                            "model_type": "torch_linear",
                            "input_type": input_type,
                            "seed": seed,
                            "threshold_strategy": cand["threshold_strategy"],
                            "mean_predicted_positive_labels": test_metrics["mean_predicted_positive_labels"],
                            "mean_true_positive_labels": test_metrics["mean_true_positive_labels"],
                        }
                    )

                # torch mlp
                mlp = MLPHead(x_train.shape[1], y_train.shape[1])
                mlp = train_torch_model(mlp, x_train, y_train, x_val, y_val, args, seed, device)
                val_logits = predict_torch(mlp, x_val, device)
                test_logits = predict_torch(mlp, x_test, device)
                candidates, _best = threshold_search(val_logits, y_val, args.threshold_strategies)
                for cand in candidates:
                    test_metrics = multilabel_metrics_np(test_logits, y_test, cand["threshold"])
                    baseline_rows.append(
                        {
                            "model_type": "torch_mlp",
                            "input_type": input_type,
                            "seed": seed,
                            "threshold_strategy": cand["threshold_strategy"],
                            "val_macro_f1": cand["val_macro_f1"],
                            "test_macro_f1": test_metrics["macro_f1"],
                            "test_micro_f1": test_metrics["micro_f1"],
                            "test_sample_f1": test_metrics["sample_f1"],
                            "mean_predicted_positive_labels": test_metrics["mean_predicted_positive_labels"],
                            "mean_true_positive_labels": test_metrics["mean_true_positive_labels"],
                            "bce_loss": mean_bce_loss_np(test_logits, y_test),
                        }
                    )
                    threshold_rows.append(
                        {
                            "model_type": "torch_mlp",
                            "input_type": input_type,
                            "seed": seed,
                            "threshold_strategy": cand["threshold_strategy"],
                            "threshold": json.dumps(np.asarray(cand["threshold"]).tolist() if isinstance(cand["threshold"], np.ndarray) else cand["threshold"]),
                            "val_macro_f1": cand["val_macro_f1"],
                            "val_micro_f1": cand["val_micro_f1"],
                            "val_sample_f1": cand["val_sample_f1"],
                            "val_bce_loss": cand["val_bce_loss"],
                        }
                    )
                    per_label_all.extend(per_label_rows("torch_mlp", input_type, seed, cand["threshold_strategy"], test_logits, y_test, cand["threshold"], class_names))
                    prediction_dist_rows.append(
                        {
                            "model_type": "torch_mlp",
                            "input_type": input_type,
                            "seed": seed,
                            "threshold_strategy": cand["threshold_strategy"],
                            "mean_predicted_positive_labels": test_metrics["mean_predicted_positive_labels"],
                            "mean_true_positive_labels": test_metrics["mean_true_positive_labels"],
                        }
                    )

        pd.DataFrame(baseline_rows).to_csv(output_dir / "baseline_results.csv", index=False)
        pd.DataFrame(threshold_rows).to_csv(output_dir / "threshold_selection.csv", index=False)
        pd.DataFrame(per_label_all).to_csv(output_dir / "per_label_f1.csv", index=False)
        pd.DataFrame(prediction_dist_rows).to_csv(output_dir / "prediction_distribution.csv", index=False)

        best_rows = pd.DataFrame(baseline_rows).sort_values("test_macro_f1", ascending=False)
        summary = [
            "# MM-IMDb Clean Multi-label Baseline Summary",
            "",
            best_rows.head(12).to_markdown(index=False),
            "",
            f"device: {device}",
            f"num_labels: {len(class_names)}",
            f"mean_true_positive_labels_test: {float(data['test']['label'].sum(axis=1).mean()):.4f}",
        ]
        write_text(output_dir / "RUN_SUMMARY.md", "\n".join(summary) + "\n")
    except Exception:
        log_error(output_dir, "run_clean_mmimdb_multilabel_baselines")
        raise


if __name__ == "__main__":
    main()
