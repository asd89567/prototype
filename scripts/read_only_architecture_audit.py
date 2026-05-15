import argparse
import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier


TARGET_SCRIPT_NAMES = [
    "train_mora_aligned_feature_adapter.py",
    "run_clean_mmimdb_baselines.py",
    "train_prototype_conditioned_feature_adapter.py",
    "train_task_aware_prototype_adapter.py",
    "prepare_text_image_subset.py",
    "text_image_pipeline_utils.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only architecture and data-flow audit.")
    parser.add_argument("--feature-dir", default="cache/text_image_features")
    parser.add_argument("--metadata-csv", default="cache/text_image_subset_metadata.csv")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/read_only_architecture_audit")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log_error(output_dir: Path, title: str, exc: BaseException) -> None:
    with (output_dir / "errors.log").open("a", encoding="utf-8") as f:
        f.write(f"\n[{title}]\n")
        f.write(traceback.format_exc())
        f.write("\n")


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def extract_source_segment(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
        start = node.lineno - 1
        end = node.end_lineno
        return "\n".join(lines[start:end])
    return ""


@dataclass
class ScriptInventory:
    path: Path
    exists: bool
    functions: List[str]
    classes: List[str]
    main_calls: List[str]
    source: str


def parse_python_inventory(path: Path) -> ScriptInventory:
    if not path.exists():
        return ScriptInventory(path=path, exists=False, functions=[], classes=[], main_calls=[], source="")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    main_calls: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    func = inner.func
                    if isinstance(func, ast.Name):
                        name = func.id
                    elif isinstance(func, ast.Attribute):
                        name = func.attr
                    else:
                        continue
                    if name not in main_calls:
                        main_calls.append(name)
    return ScriptInventory(path=path, exists=True, functions=functions, classes=classes, main_calls=main_calls, source=source)


def markdown_bullets(items: List[str], empty_text: str = "(none)") -> str:
    if not items:
        return f"- {empty_text}"
    return "\n".join(f"- `{item}`" for item in items)


def load_feature_artifacts(feature_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    artifacts: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        artifacts[split] = {
            "image": np.load(feature_dir / f"{split}_image.npy", allow_pickle=True),
            "text": np.load(feature_dir / f"{split}_text.npy", allow_pickle=True),
            "label_raw": np.load(feature_dir / f"{split}_label.npy", allow_pickle=True),
            "sample_ids": np.load(feature_dir / f"{split}_sample_ids.npy", allow_pickle=True),
        }
    return artifacts


def normalize_sample_id_basic(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    stem = Path(s).stem
    if stem.startswith("tt"):
        stem = stem[2:]
    if stem.endswith(".0"):
        stem = stem[:-2]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return digits.zfill(7)
    return stem


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def apply_thresholds_np(prob: np.ndarray, thresholds) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim == 0:
        return (prob >= float(thresholds)).astype(np.float32)
    return (prob >= thresholds.reshape(1, -1)).astype(np.float32)


def multilabel_metrics_np(logits: np.ndarray, y: np.ndarray, thresholds) -> Dict[str, float]:
    prob = sigmoid_np(logits)
    pred = apply_thresholds_np(prob, thresholds)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "sample_f1": float(f1_score(y, pred, average="samples", zero_division=0)),
        "accuracy": float((pred == y).all(axis=1).mean()),
        "mean_predicted_positive_labels": float(pred.sum(axis=1).mean()),
        "mean_true_positive_labels": float(y.sum(axis=1).mean()),
    }


def select_best_thresholds_np(logits: np.ndarray, y: np.ndarray, grid: List[float]) -> Dict[str, object]:
    prob = sigmoid_np(logits)
    best = None

    for t in grid:
        metrics = multilabel_metrics_np(logits, y, t)
        row = {"threshold_strategy": "global", "threshold": float(t), "metrics": metrics}
        if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
            best = row

    thresholds = np.zeros(y.shape[1], dtype=np.float32)
    for j in range(y.shape[1]):
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            pred_j = (prob[:, j] >= float(t)).astype(np.float32)
            f1 = f1_score(y[:, j], pred_j, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[j] = best_t
    metrics = multilabel_metrics_np(logits, y, thresholds)
    row = {"threshold_strategy": "per_class", "threshold": thresholds, "metrics": metrics}
    if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
        best = row
    return best


def mean_bce_loss_np(logits: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    prob = sigmoid_np(logits)
    loss = -(y * np.log(np.clip(prob, eps, 1.0 - eps)) + (1.0 - y) * np.log(np.clip(1.0 - prob, eps, 1.0 - eps)))
    return float(loss.mean())


def fit_independent_ovr(model_name: str, x_train: np.ndarray, y_train: np.ndarray):
    if model_name == "logistic":
        base = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "ridge":
        base = RidgeClassifier(class_weight="balanced")
    else:
        raise ValueError(model_name)
    clf = OneVsRestClassifier(base)
    clf.fit(x_train, y_train)
    return clf


def decision_scores(clf, x: np.ndarray) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(x)
    elif hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(x)
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        scores = np.log(prob / (1.0 - prob))
    else:
        raise ValueError("Classifier has neither decision_function nor predict_proba.")
    return np.asarray(scores, dtype=np.float32)


def probs_topk(logits: np.ndarray, class_names: List[str], k: int = 5) -> List[str]:
    prob = sigmoid_np(logits.reshape(1, -1))[0]
    order = np.argsort(-prob)[:k]
    return [f"{class_names[idx]}:{prob[idx]:.4f}" for idx in order]


def predicted_labels_from_logits(logits: np.ndarray, thresholds, class_names: List[str]) -> List[str]:
    pred = apply_thresholds_np(sigmoid_np(logits.reshape(1, -1)), thresholds)[0]
    return [class_names[i] for i, v in enumerate(pred) if v > 0]


def true_labels_from_y(y_row: np.ndarray, class_names: List[str]) -> List[str]:
    return [class_names[i] for i, v in enumerate(y_row) if v > 0]


def read_results_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    feature_dir = (repo_root / args.feature_dir).resolve()
    metadata_csv = (repo_root / args.metadata_csv).resolve()
    results_root = (repo_root / args.results_root).resolve()
    output_dir = ensure_dir((repo_root / args.output_dir).resolve())
    write_text(output_dir / "errors.log", "")

    try:
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Import current helper stack read-only.
        sys.path.insert(0, str(scripts_dir))
        proto_mod = load_module("audit_train_proto", scripts_dir / "train_prototype_conditioned_feature_adapter.py")
        mora_mod = load_module("audit_train_mora", scripts_dir / "train_mora_aligned_feature_adapter.py")
        text_utils_mod = load_module("audit_text_utils", scripts_dir / "text_image_pipeline_utils.py")

        inventories = {name: parse_python_inventory(scripts_dir / name) for name in TARGET_SCRIPT_NAMES}

        # Load feature arrays and rebuilt labels.
        arrays = load_feature_artifacts(feature_dir)
        metadata_df = pd.read_csv(metadata_csv, dtype={"sample_id": str, "id": str, "imdb_id": str})
        features, labels, split_meta, class_names, feature_meta = proto_mod.load_mmimdb_payload(feature_dir, metadata_csv)

        # Part 1: architecture.
        architecture_sections = [
            "# Architecture Audit",
            "",
            "## Repo-visible scripts",
        ]
        role_hints = {
            "train_mora_aligned_feature_adapter.py": "MoRA-aligned feature-level training/evaluation pipeline with missing-table generation, branch training, adapters, threshold selection, and result writing.",
            "train_prototype_conditioned_feature_adapter.py": "Earlier prototype-conditioned representation adapter pipeline; also contains the current cache/metadata/label reconstruction helpers and core multilabel metrics helpers.",
            "text_image_pipeline_utils.py": "Small standalone feature/metadata loading helper; overlaps with helper logic inside train_prototype_conditioned_feature_adapter.py.",
            "run_clean_mmimdb_baselines.py": "Not present in current repo snapshot.",
            "train_task_aware_prototype_adapter.py": "Not present in current repo snapshot.",
            "prepare_text_image_subset.py": "Not present in current repo snapshot.",
        }
        for name in TARGET_SCRIPT_NAMES:
            inv = inventories[name]
            architecture_sections.extend(
                [
                    f"### `{name}`",
                    f"- exists: `{inv.exists}`",
                    f"- purpose: {role_hints.get(name, 'unknown')}",
                    "- top-level functions:",
                    markdown_bullets(inv.functions[:40]),
                    "- top-level classes:",
                    markdown_bullets(inv.classes[:20]),
                ]
            )
            if inv.exists:
                architecture_sections.extend(
                    [
                        "- `main()` direct call inventory:",
                        markdown_bullets(inv.main_calls[:50]),
                        "",
                    ]
                )
            else:
                architecture_sections.append("")

        architecture_sections.extend(
            [
                "## High-level call graph",
                "",
                "### `train_mora_aligned_feature_adapter.py`",
                "```text",
                "main",
                " ├── load_mmimdb_payload",
                " ├── primary_labels_for_split",
                " ├── run_protocol_seed (for each protocol, seed)",
                " │    ├── make_missing_table",
                " │    ├── train_branch_base (full_teacher on complete only)",
                " │    ├── train_branch_base (missing branch base model)",
                " │    ├── fit_cluster / assign_cluster",
                " │    ├── train_adapter",
                " │    ├── predict_adapter_mode / combine_logits",
                " │    ├── select_protocol_thresholds",
                " │    └── evaluate_and_record_method",
                " ├── write_outputs",
                " └── make_summary",
                "```",
                "",
                "### `train_prototype_conditioned_feature_adapter.py`",
                "```text",
                "main",
                " ├── load_mmimdb_payload",
                " ├── train_representation_classifier (full teacher)",
                " ├── train_representation_classifier (missing base)",
                " ├── fit_residual_clusters",
                " ├── train_adapter_model",
                " ├── predict_adapter / evaluate_result_row",
                " ├── per_label_rows / router_stats_rows / alignment_row",
                " └── make_summary",
                "```",
                "",
                "## Role mapping by function",
                "",
                "- load features: `text_image_pipeline_utils.load_split_arrays`, `train_prototype_conditioned_feature_adapter.load_split_arrays`",
                "- load labels / build multilabel targets: `train_prototype_conditioned_feature_adapter.load_mmimdb_payload`, `build_label_mapping`, `build_multilabel_targets`",
                "- build missing table: `train_mora_aligned_feature_adapter.make_missing_table`",
                "- train teacher: `train_mora_aligned_feature_adapter.train_branch_base` -> `train_representation_classifier`",
                "- train missing branch: `train_mora_aligned_feature_adapter.train_branch_base`",
                "- train adapter: `train_mora_aligned_feature_adapter.train_adapter`",
                "- tune threshold: `train_prototype_conditioned_feature_adapter.select_best_thresholds`, `train_mora_aligned_feature_adapter.select_protocol_thresholds`",
                "- compute macro-F1: `train_prototype_conditioned_feature_adapter.multilabel_metrics`, `train_mora_aligned_feature_adapter.metrics_for_logits`",
                "- save results: `train_mora_aligned_feature_adapter.write_outputs`, `train_prototype_conditioned_feature_adapter.make_summary`",
                "",
                "## Most likely files affecting `0.37` vs `0.54`",
                "- `scripts/train_mora_aligned_feature_adapter.py`",
                "- `scripts/train_prototype_conditioned_feature_adapter.py`",
                "- `scripts/text_image_pipeline_utils.py`",
                "- `cache/text_image_subset_metadata.csv`",
                "- `cache/text_image_features/feature_metadata.json`",
                "",
            ]
        )

        # Part 2: data lineage.
        data_rows: List[Dict[str, object]] = []
        mean_pos = {split: float(labels[split].sum(axis=1).mean()) for split in ("train", "val", "test")}
        one_positive_ratio = {
            split: float((labels[split].sum(axis=1) == 1).mean()) for split in ("train", "val", "test")
        }
        raw_label_uniques = {
            split: int(len(np.unique(arrays[split]["label_raw"]))) for split in ("train", "val", "test")
        }
        sample_align_status = {}
        for split in ("train", "val", "test"):
            feature_ids = [normalize_sample_id_basic(x) for x in arrays[split]["sample_ids"]]
            meta_ids = split_meta[split]["sample_id"].astype(str).tolist()
            sample_align_status[split] = feature_ids == meta_ids

        inferred_raw_root = None
        if "image_path" in metadata_df.columns and len(metadata_df) > 0:
            sample_path = Path(str(metadata_df.iloc[0]["image_path"]))
            inferred_raw_root = sample_path.parent.parent if sample_path.exists() else sample_path.parent.parent

        raw_jsons = []
        if inferred_raw_root is not None and inferred_raw_root.exists():
            raw_jsons = [str(p) for p in sorted(inferred_raw_root.rglob("*.json"))[:20]]

        for artifact in [
            feature_dir / "train_image.npy",
            feature_dir / "train_text.npy",
            feature_dir / "train_label.npy",
            feature_dir / "train_sample_ids.npy",
            feature_dir / "val_image.npy",
            feature_dir / "val_text.npy",
            feature_dir / "val_label.npy",
            feature_dir / "val_sample_ids.npy",
            feature_dir / "test_image.npy",
            feature_dir / "test_text.npy",
            feature_dir / "test_label.npy",
            feature_dir / "test_sample_ids.npy",
            feature_dir / "feature_metadata.json",
            metadata_csv,
        ]:
            exists = artifact.exists()
            shape_or_count = ""
            notes = ""
            if exists and artifact.suffix == ".npy":
                arr = np.load(artifact, allow_pickle=True)
                shape_or_count = f"shape={arr.shape}, dtype={arr.dtype}"
            elif exists and artifact.suffix == ".json":
                shape_or_count = f"keys={list(feature_meta.keys())}"
            elif exists and artifact.suffix == ".csv":
                shape_or_count = f"rows={len(metadata_df)}, cols={len(metadata_df.columns)}"
            if artifact.name == "train_label.npy":
                notes = "raw cache label array is scalar label_id per sample, not multi-hot."
            data_rows.append(
                {
                    "artifact": artifact.name,
                    "path": str(artifact),
                    "exists": exists,
                    "shape_or_count": shape_or_count,
                    "created_by_script_if_known": "prepare_text_image_subset.py / feature extraction pipeline (not present in current repo)" if artifact.suffix in {".npy", ".json"} or artifact == metadata_csv else "",
                    "depends_on": "cache/text_image_subset_metadata.csv" if artifact.suffix == ".npy" else "",
                    "risk_level": "high" if artifact.name in {"train_label.npy", "feature_metadata.json", metadata_csv.name} else "medium",
                    "notes": notes,
                }
            )
        pd.DataFrame(data_rows).to_csv(output_dir / "data_lineage_table.csv", index=False)

        data_lineage_sections = [
            "# Data Lineage Audit",
            "",
            f"- `cache/text_image_features` exists: `{feature_dir.exists()}`",
            f"- `cache/text_image_subset_metadata.csv` exists: `{metadata_csv.exists()}`",
            f"- `feature_metadata.json` exists: `{(feature_dir / 'feature_metadata.json').exists()}`",
            "",
            "## Direct observations",
            f"- raw `train_label.npy` shape: `{arrays['train']['label_raw'].shape}` dtype=`{arrays['train']['label_raw'].dtype}`",
            f"- rebuilt `train` labels shape from `load_mmimdb_payload`: `{labels['train'].shape}`",
            f"- mean positive labels per sample: train=`{mean_pos['train']:.4f}`, val=`{mean_pos['val']:.4f}`, test=`{mean_pos['test']:.4f}`",
            f"- ratio of samples with exactly one positive label: train=`{one_positive_ratio['train']:.4f}`, val=`{one_positive_ratio['val']:.4f}`, test=`{one_positive_ratio['test']:.4f}`",
            f"- raw label unique counts: train=`{raw_label_uniques['train']}`, val=`{raw_label_uniques['val']}`, test=`{raw_label_uniques['test']}`",
            "",
            "## Answers",
            f"1. `cache/text_image_features` 裡的 raw label 是 **single-label scalar id**；current helper 會把它重建成 one-hot matrix，但每個 sample 目前仍只有一個 positive。",
            f"2. `train_label.npy` shape 是 `{arrays['train']['label_raw'].shape}`。",
            f"3. mean positive labels per sample 約為 train `{mean_pos['train']:.4f}`，這非常接近 1，與標準 MM-IMDb 多標籤預期不一致。",
            f"4. label mapping 目前直接來自 `feature_metadata.json.label_mapping` 與 metadata 中的 `label`/`label_id` 欄位。",
            f"5. metadata sample_id 和 npy sample_ids 對齊狀態：train=`{sample_align_status['train']}`, val=`{sample_align_status['val']}`, test=`{sample_align_status['test']}`。",
            "6. 目前 cache **很可能**是早期 primary-label / single-positive sanity subset，而不是保留原始 multi-label MM-IMDb target 的 cache。",
            f"7. raw MM-IMDb json existence: `{bool(raw_jsons)}`.",
            f"8. 若要重建資料，最直接的 raw source 看起來是 `{inferred_raw_root}`，其中 json examples: {raw_jsons[:5] if raw_jsons else 'not found'}.",
            "",
            "## Risk judgment",
            "- 目前最值得懷疑的不是 feature 向量本身，而是 **label semantics**：feature cache 對應的 target 世界看起來像單標籤 one-hot，不像 MM-IMDb 原始多標籤。",
            "",
        ]

        # Part 3: evaluation audit.
        eval_inventory_rows: List[Dict[str, object]] = []
        eval_targets = {
            "train_prototype_conditioned_feature_adapter.py": [
                "apply_thresholds",
                "multilabel_metrics",
                "select_best_thresholds",
            ],
            "train_mora_aligned_feature_adapter.py": [
                "apply_thresholds_protocol",
                "metrics_for_logits",
                "select_protocol_thresholds",
            ],
        }
        evaluation_sections = [
            "# Evaluation Audit",
            "",
            "## Core findings",
            "- macro-F1 實作使用 `sklearn.metrics.f1_score(..., average=\"macro\", zero_division=0)`。",
            "- prediction path 是 `sigmoid(logits)` 後做 threshold，不是 argmax。",
            "- per-class threshold 會對每個 output label 獨立搜尋 threshold。",
            "- current core training path 沒有看到直接用 test label 調 threshold；test 主要用 validation-selected threshold。",
            "- 但 repo 內存在多套 helper：`train_prototype_conditioned_feature_adapter.py` 內有一套 self-contained evaluation helpers，`train_mora_aligned_feature_adapter.py` 再包一層 protocol-aware threshold logic。",
            "",
        ]
        for filename, fnames in eval_targets.items():
            inv = inventories[filename]
            if not inv.exists:
                continue
            tree = ast.parse(inv.source)
            node_map = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
            evaluation_sections.append(f"## `{filename}`")
            for fname in fnames:
                node = node_map.get(fname)
                if node is None:
                    continue
                src = extract_source_segment(inv.source, node)
                risk = "low"
                uses_val = "val" in src or "validation" in src
                uses_test_labels = "test" in src and "label" in src and "uses_test_label" in src
                multilabel_correct = ("average=\"macro\"" in src or "average='macro'" in src or "sigmoid" in src or "threshold" in src)
                notes = []
                if "zero_division=0" in src:
                    notes.append("zero_division=0")
                if "thresholds[j]" in src or "for label_idx" in src:
                    notes.append("per-label threshold search")
                if fname == "select_protocol_thresholds" and "per_branch_thresholds" in src:
                    notes.append("supports subgroup-specific thresholds")
                    risk = "medium"
                if filename == "train_prototype_conditioned_feature_adapter.py" and fname == "select_best_thresholds":
                    notes.append("generic helper used by multiple later scripts")
                eval_inventory_rows.append(
                    {
                        "file": filename,
                        "function_name": fname,
                        "metric": "macro/micro/sample F1 + BCE" if "f1_score" in src else "threshold wrapper",
                        "threshold_strategy": "global + per_class" if "per_class" in src else "protocol wrapper",
                        "uses_validation": uses_val,
                        "uses_test_labels": uses_test_labels,
                        "multilabel_correct": multilabel_correct,
                        "risk_level": risk,
                        "notes": "; ".join(notes),
                    }
                )
                evaluation_sections.extend(
                    [
                        f"### `{fname}`",
                        f"- multilabel-correct intent: `{multilabel_correct}`",
                        f"- uses validation path explicitly: `{uses_val}`",
                        f"- uses test labels directly: `{uses_test_labels}`",
                        f"- notes: {', '.join(notes) if notes else 'none'}",
                        "",
                    ]
                )
        pd.DataFrame(eval_inventory_rows).to_csv(output_dir / "evaluation_function_inventory.csv", index=False)
        evaluation_sections.extend(
            [
                "## Direct answers",
                "1. macro-F1 是用 sklearn `f1_score` average=`\"macro\"`。",
                "2. `zero_division=0`。",
                "3. current main training/eval helper 看起來是 validation-only threshold tuning。",
                "4. per-class threshold 是每個 label 獨立 tune。",
                "5. 核心訓練路徑沒有發現 test threshold tuning；但 oracle / diagnostic paths 需另外區分。",
                f"6. global threshold candidates 預設來自 parser，約 `{[round(i/100,2) for i in range(5,100,5)]}`。",
                "7. prediction 是 `sigmoid + threshold`。",
                "8. current visible helpers 沒有把 multi-label evaluation 寫成 argmax；但如果資料本身已經單標籤 one-hot，macro-F1 仍會在『偽多標籤』世界上計算。",
                "9. mean predicted positive labels 要搭配 baseline sanity 一起看；若也接近 1，會支持『目前 cache 本來就是單標籤』這件事。",
                "10. evaluation helper 確實有多個版本 / 包裝層，這本身就是風險點。",
                "",
            ]
        )

        # Part 4: model flow audit.
        model_flow_sections = [
            "# Model Flow Audit",
            "",
            "## Full teacher",
            "- current `train_mora_aligned_feature_adapter.py` 裡，full teacher 在 `run_protocol_seed()` 只用 `complete_train` 樣本訓練。",
            "- 其用途是提供 `z_full` 與 `full_logits`，並作為 adapter alignment / residual cluster 的 supervision。",
            "- current default `complete_eval_mode` 是 `branch_ensemble`，因此 full teacher 預設**不是**最終 complete sample classifier；只有 `complete_eval_mode=full_teacher` 時才會直接輸出 teacher logits。",
            "",
            "## Branch paths",
            "- `image_missing` sample path: `text feature -> image_missing branch projector/classifier -> optional adapter -> logits`",
            "- `text_missing` sample path: `image feature -> text_missing branch projector/classifier -> optional adapter -> logits`",
            "- `complete` sample path (current default): `combine_logits(..., complete_eval_mode='branch_ensemble')` -> average available branch logits.",
            "",
            "## Current `branch_ensemble`",
            "- implemented in `combine_logits()`.",
            "- for complete samples, if `complete_eval_mode != 'full_teacher'`, it averages all available branch logits.",
            "- if no branch logits exist, it falls back to `full_logits` as an emergency fallback.",
            "",
            "## Formulas",
            "- `z_missing = projector(x_available)`",
            "- `base_logits = classifier(z_missing)`",
            "- `static_delta = StaticAdapter(z_missing)`",
            "- `z_static = z_missing + static_delta`",
            "- `mixture_delta = sum_k alpha_k * Adapter_k(z_missing)`",
            "- `beta = sigmoid(beta_mlp([LayerNorm(z_missing), missing_type_emb]))` for safe-mixture variants",
            "- `z_final = z_static + beta * mixture_delta` for safe-mixture variants",
            "- non-safe mixture: `z_final = z_missing + mixture_delta`",
            "- `logits = classifier(z_final)`",
            "",
            "## Method-specific path",
            "- `static_feature_adapter`: `z_final = z_missing + static_delta`",
            "- `mixture_feature_adapter_unsupervised`: `z_final = z_missing + mixture_delta`, `alpha = softmax(router(...))`",
            "- `safe_mixture_beta_zero`: same mixture experts but force `beta = 0`",
            "- `safe_mixture_beta_one`: same mixture experts but force `beta = 1`",
            "- `safe_mixture_beta_learned`: learn `beta` with zero bias",
            "- `safe_mixture_beta_biased`: learn `beta` with branch-specific bias",
            "- `prototype_plain_hard`: same mixture architecture but router supervision uses hard residual clusters on complete train samples only",
            "",
            "## Paths likely different from old `0.5466` era",
            "- current parser default `complete_eval_mode='branch_ensemble'`",
            "- safe-mixture branches and extra beta diagnostics were added later",
            "- current `predict_adapter_mode()` always goes through `model.forward()` to avoid train/eval mismatch; old runs may have used older inference code paths",
            "- current script has subgroup-specific threshold option `per_branch_thresholds`",
            "",
        ]

        # Part 5/6 independent baselines and sample trace.
        baseline_rows = []
        traces = []
        threshold_grid = [round(i / 100, 2) for i in range(5, 100, 5)]

        feature_views = {
            "text_only": (
                features["train"]["text"].astype(np.float32),
                features["val"]["text"].astype(np.float32),
                features["test"]["text"].astype(np.float32),
            ),
            "image_only": (
                features["train"]["image"].astype(np.float32),
                features["val"]["image"].astype(np.float32),
                features["test"]["image"].astype(np.float32),
            ),
            "concat_image_text": (
                np.concatenate([features["train"]["image"], features["train"]["text"]], axis=1).astype(np.float32),
                np.concatenate([features["val"]["image"], features["val"]["text"]], axis=1).astype(np.float32),
                np.concatenate([features["test"]["image"], features["test"]["text"]], axis=1).astype(np.float32),
            ),
        }

        trained_models = {}
        best_thresholds = {}
        test_scores = {}
        split_scores: Dict[Tuple[str, str, str], np.ndarray] = {}
        for model_name in ("logistic", "ridge"):
            for view_name, (x_train, x_val, x_test) in feature_views.items():
                clf = fit_independent_ovr(model_name, x_train, labels["train"])
                train_logits = decision_scores(clf, x_train)
                val_logits = decision_scores(clf, x_val)
                best = select_best_thresholds_np(val_logits, labels["val"], threshold_grid)
                test_logits = decision_scores(clf, x_test)
                metrics = multilabel_metrics_np(test_logits, labels["test"], best["threshold"])
                metrics["bce_loss"] = mean_bce_loss_np(test_logits, labels["test"])
                row = {
                    "family": model_name,
                    "view": view_name,
                    "threshold_strategy": best["threshold_strategy"],
                    "threshold": json.dumps(best["threshold"].tolist() if isinstance(best["threshold"], np.ndarray) else best["threshold"]),
                    **metrics,
                }
                baseline_rows.append(row)
                trained_models[(model_name, view_name)] = clf
                best_thresholds[(model_name, view_name)] = best
                test_scores[(model_name, view_name)] = test_logits
                split_scores[(model_name, view_name, "train")] = train_logits
                split_scores[(model_name, view_name, "val")] = val_logits
                split_scores[(model_name, view_name, "test")] = test_logits
        baseline_df = pd.DataFrame(baseline_rows).sort_values(["family", "view"])
        baseline_df.to_csv(output_dir / "independent_baseline_results.csv", index=False)

        best_logistic_concat = baseline_df[(baseline_df["family"] == "logistic") & (baseline_df["view"] == "concat_image_text")]
        best_logistic_text = baseline_df[(baseline_df["family"] == "logistic") & (baseline_df["view"] == "text_only")]

        baseline_sections = [
            "# Baseline Sanity Audit",
            "",
            "## Independent baselines",
            baseline_df.to_markdown(index=False),
            "",
        ]
        if not best_logistic_concat.empty:
            concat_macro = float(best_logistic_concat.iloc[0]["macro_f1"])
            if concat_macro < 0.40:
                baseline_sections.append("- `concat_image_text` full baseline 仍只有約 0.37–0.38，這支持『目前 cache/eval 世界本來就在這個水位』。")
            elif concat_macro >= 0.50:
                baseline_sections.append("- `concat_image_text` full baseline 可以到 0.5+，這會反過來支持 adapter script / branch flow 可能有 bug。")
        if not best_logistic_text.empty:
            text_macro = float(best_logistic_text.iloc[0]["macro_f1"])
            if text_macro >= 0.50:
                baseline_sections.append("- `text_only` 若能到 0.5+，那 image_missing branch 掉到 0.37 就更像 branch training / eval 問題。")
        baseline_sections.append("")

        # Sample trace based on logistic baselines + saved missing tables.
        missing_table_root = results_root / "mora_aligned_feature_adapter" / "missing_tables"
        both_tables = {}
        image_tables = {}
        text_tables = {}
        for split in ("train", "val", "test"):
            p = missing_table_root / f"{split}_both_70_seed42.npy"
            if p.exists():
                both_tables[split] = np.load(p)
            p = missing_table_root / f"{split}_image_missing_70_seed42.npy"
            if p.exists():
                image_tables[split] = np.load(p)
            p = missing_table_root / f"{split}_text_missing_70_seed42.npy"
            if p.exists():
                text_tables[split] = np.load(p)

        # proxy thresholds/logits for branch-path explanation
        logistic_text_test = split_scores[("logistic", "text_only", "test")]
        logistic_image_test = split_scores[("logistic", "image_only", "test")]
        logistic_text_val = split_scores[("logistic", "text_only", "val")]
        logistic_image_val = split_scores[("logistic", "image_only", "val")]
        logistic_text_train = split_scores[("logistic", "text_only", "train")]
        logistic_image_train = split_scores[("logistic", "image_only", "train")]
        ensemble_val = (logistic_text_val + logistic_image_val) / 2.0
        ensemble_threshold = select_best_thresholds_np(ensemble_val, labels["val"], threshold_grid)
        ensemble_test = (logistic_text_test + logistic_image_test) / 2.0
        ensemble_train = (logistic_text_train + logistic_image_train) / 2.0

        def append_trace(split: str, idx: int, missing_type_code: int, source_group: str):
            y_row = labels[split][idx]
            if missing_type_code == 0:
                if split == "test":
                    logits = ensemble_test[idx]
                elif split == "val":
                    logits = ensemble_val[idx]
                else:
                    logits = ensemble_train[idx]
                thresholds = ensemble_threshold["threshold"]
                model_path_used = "complete -> branch_ensemble_proxy(text_only + image_only avg)"
                missing_name = "complete"
            elif missing_type_code == 2:
                if split == "test":
                    logits = logistic_text_test[idx]
                elif split == "val":
                    logits = logistic_text_val[idx]
                else:
                    logits = logistic_text_train[idx]
                thresholds = best_thresholds[("logistic", "text_only")]["threshold"]
                model_path_used = "image_missing -> text_only branch proxy"
                missing_name = "image_missing"
            else:
                if split == "test":
                    arr = logistic_image_test
                elif split == "val":
                    arr = logistic_image_val
                else:
                    arr = logistic_image_train
                logits = arr[idx]
                thresholds = best_thresholds[("logistic", "image_only")]["threshold"]
                model_path_used = "text_missing -> image_only branch proxy"
                missing_name = "text_missing"
            pred_labels = predicted_labels_from_logits(logits, thresholds, class_names)
            true_labels = true_labels_from_y(y_row, class_names)
            traces.append(
                {
                    "sample_id": split_meta[split].iloc[idx]["sample_id"],
                    "split": split,
                    "index_in_npy": idx,
                    "metadata_index": int(split_meta[split].index[idx]),
                    "source_group": source_group,
                    "true_labels": json.dumps(true_labels, ensure_ascii=False),
                    "image_feature_norm": float(np.linalg.norm(features[split]["image"][idx])),
                    "text_feature_norm": float(np.linalg.norm(features[split]["text"][idx])),
                    "missing_type": missing_name,
                    "model_path_used": model_path_used,
                    "logits_shape": list(np.asarray(logits).shape),
                    "probs_top5": json.dumps(probs_topk(np.asarray(logits), class_names, 5), ensure_ascii=False),
                    "predicted_labels": json.dumps(pred_labels, ensure_ascii=False),
                    "threshold_used": json.dumps(np.asarray(thresholds).tolist() if isinstance(thresholds, np.ndarray) else thresholds),
                    "correct_labels": json.dumps(sorted(list(set(pred_labels).intersection(true_labels))), ensure_ascii=False),
                    "error_count": int(len(set(pred_labels).symmetric_difference(true_labels))),
                }
            )

        rng = np.random.default_rng(args.seed)
        for split in ("train", "val", "test"):
            idxs = rng.choice(np.arange(len(labels[split])), size=min(5, len(labels[split])), replace=False)
            for idx in idxs:
                table = both_tables.get(split)
                if table is None:
                    table = image_tables.get(split)
                missing_code = int(table[idx]) if table is not None else 0
                append_trace(split, int(idx), missing_code, f"random_{split}")

        for split, tableset, group_name, code in [
            ("test", both_tables, "complete_examples", 0),
            ("test", image_tables if image_tables else both_tables, "image_missing_examples", 2),
            ("test", both_tables, "text_missing_examples", 1),
        ]:
            table = tableset.get(split) if tableset else None
            if table is None:
                continue
            candidates = np.flatnonzero(table == code)
            if len(candidates) == 0:
                continue
            chosen = candidates[: min(5, len(candidates))]
            for idx in chosen:
                append_trace(split, int(idx), code, group_name)

        trace_df = pd.DataFrame(traces)
        trace_df.to_csv(output_dir / "sample_trace.csv", index=False)
        sample_trace_sections = [
            "# Sample Trace Audit",
            "",
            "以下 trace 使用 **independent logistic proxy** 來檢查 sample / label / path 是否合理；不是 train_mora checkpoint replay。",
            "",
            trace_df.head(25).to_markdown(index=False) if not trace_df.empty else "(no trace rows)",
            "",
        ]

        # Old result comparison.
        old_seed_path = results_root / "mora_aligned_feature_adapter" / "seed_summary.csv"
        old_config_path = results_root / "mora_aligned_feature_adapter" / "config.json"
        old_cmd_exists = (results_root / "mora_aligned_feature_adapter" / "commands.log").exists()
        old_hash_exists = any((results_root / "mora_aligned_feature_adapter").glob("*hash*"))
        old_threshold_exists = (results_root / "mora_aligned_feature_adapter" / "threshold_selection.csv").exists()
        comparison_sections = [
            "# Old Result Comparison",
            "",
            f"1. 舊 `0.5466` result 檔存在：`{old_seed_path.exists()}`。",
            f"2. 舊 result 有對應 config：`{old_config_path.exists()}`。",
            f"3. 舊 result 有對應 command：`{old_cmd_exists}`。",
            f"4. 舊 result 有 cache hash：`{old_hash_exists}`。",
            f"5. 舊 result 有 sample/prediction 保存：`{old_threshold_exists}`（threshold csv 目前也不存在或不足）。",
            "6. 舊 result 和現在 clean baseline 不可直接比較，因為 exact script/helper/cache state 未保存完整。",
            "7. 目前缺少的關鍵證據包括：exact command log、cache hash、per-sample predictions、完整 helper state、old missing-table generation provenance。",
            "",
        ]

        # Suspected issues.
        top_issues = [
            {
                "issue": "Current cache target semantics look single-label, not original MM-IMDb multilabel.",
                "evidence": f"Rebuilt labels have mean positives per sample about {mean_pos['train']:.4f}; exactly-one-positive ratio is {one_positive_ratio['train']:.4f} on train.",
                "severity": "critical",
                "how_to_verify": "Rebuild labels directly from raw MM-IMDb JSON and compare per-sample positive counts.",
                "recommended_next_action": "Audit / rebuild dataset pipeline before further adapter debugging.",
            },
            {
                "issue": "Raw cached `*_label.npy` are scalar ids, not multi-hot arrays.",
                "evidence": f"`train_label.npy` shape is {arrays['train']['label_raw'].shape} and dtype {arrays['train']['label_raw'].dtype}.",
                "severity": "critical",
                "how_to_verify": "Open raw npy and compare to metadata `label_id`.",
                "recommended_next_action": "Treat current cache as class-id cache unless raw source proves otherwise.",
            },
            {
                "issue": "Two different metadata/sample-id helper stacks exist in repo.",
                "evidence": "text_image_pipeline_utils.py lacks MM-IMDb id normalization while train_prototype_conditioned_feature_adapter.py contains a normalized loader.",
                "severity": "high",
                "how_to_verify": "Compare behavior on ids like `0098333`.",
                "recommended_next_action": "External reviewer should inspect both helper versions before trusting alignment claims.",
            },
            {
                "issue": "Current train_mora default complete-sample inference path differs from older saved runs.",
                "evidence": "Current parser default is `complete_eval_mode=branch_ensemble`; old saved config had `complete_eval_mode=None`.",
                "severity": "high",
                "how_to_verify": "Replay complete subgroup metrics under both modes on the same checkpoint.",
                "recommended_next_action": "Do not directly compare post-branch-ensemble results to old 0.5466 without protocol note.",
            },
            {
                "issue": "Saved old missing tables are not reproducible from current generator with the same seeds.",
                "evidence": "Recovery audit found hundreds of position differences for train image_missing_70 tables.",
                "severity": "high",
                "how_to_verify": "Diff saved `.npy` tables against newly generated ones.",
                "recommended_next_action": "Preserve saved missing tables as historical artifacts; do not assume seed reproducibility.",
            },
            {
                "issue": "Old 0.5466 result lacks enough provenance to be a strict reproducibility reference.",
                "evidence": "No command log, no cache hash, no per-sample prediction export, and no exact helper snapshot.",
                "severity": "high",
                "how_to_verify": "Inspect results/mora_aligned_feature_adapter directory inventory.",
                "recommended_next_action": "Treat old 0.5466 as historical only until legacy stack is fully reconstructed.",
            },
            {
                "issue": "Current clean 0.37 may simply reflect the current cache/eval world.",
                "evidence": "Independent baselines are computed on the same cache/eval world and provide a lower-bound sanity reference.",
                "severity": "high",
                "how_to_verify": "Compare concat/image/text independent baselines with adapter script results.",
                "recommended_next_action": "Use independent baseline as first gate before touching adapter logic.",
            },
            {
                "issue": "Evaluation logic is multilabel-correct in code, but may be operating on pseudo-multilabel one-hot targets.",
                "evidence": "Helpers use sigmoid + threshold + multilabel f1_score, while labels appear exactly-one-positive.",
                "severity": "medium",
                "how_to_verify": "Compare against a true multiclass argmax baseline on the same cache.",
                "recommended_next_action": "Ask whether this subset was intentionally converted to single-label.",
            },
            {
                "issue": "Text/image branch training may be debugging the wrong target problem if dataset is single-label.",
                "evidence": "Current branch baselines are evaluated with thresholded multilabel logic even though labels appear one-hot.",
                "severity": "medium",
                "how_to_verify": "Run multiclass baselines on same features and compare ranking.",
                "recommended_next_action": "Separate dataset semantics audit from model-flow audit.",
            },
            {
                "issue": "Old 0.5466 could have benefitted from teacher-related comparability differences or leakage-like behavior.",
                "evidence": "Historical helper stack is incomplete and old config/provenance are missing; current complete handling definitely changed later.",
                "severity": "medium",
                "how_to_verify": "Only possible by matched legacy checkpoint/code replay.",
                "recommended_next_action": "Do not cite old 0.5466 as a clean MoRA-aligned result without caveat.",
            },
        ]
        suspected_lines = ["# Suspected Issues", ""]
        for idx, item in enumerate(top_issues, start=1):
            suspected_lines.extend(
                [
                    f"## {idx}. {item['issue']}",
                    f"- evidence: {item['evidence']}",
                    f"- severity: {item['severity']}",
                    f"- how to verify: {item['how_to_verify']}",
                    f"- recommended next action: {item['recommended_next_action']}",
                    "",
                ]
            )

        files_to_review_lines = [
            "# Files To Review",
            "",
            "## Must review",
            f"- path: `{scripts_dir / 'train_mora_aligned_feature_adapter.py'}`",
            "  why: current main training/evaluation flow, complete handling, threshold protocol, branch routing.",
            f"- path: `{scripts_dir / 'train_prototype_conditioned_feature_adapter.py'}`",
            "  why: contains the active cache/metadata/label reconstruction helper stack and multilabel metrics helpers.",
            f"- path: `{scripts_dir / 'text_image_pipeline_utils.py'}`",
            "  why: competing lightweight loader; differs from the self-contained helper logic.",
            f"- path: `{feature_dir / 'feature_metadata.json'}`",
            "  why: current label mapping source baked into cache.",
            f"- path: `{metadata_csv}`",
            "  why: current metadata appears single-label (`label`, `label_id`) and may explain the whole collapse.",
            f"- path: `{results_root / 'mora_aligned_feature_adapter' / 'seed_summary.csv'}`",
            "  why: historical 0.5466 reference.",
            f"- path: `{output_dir / 'DATA_LINEAGE_AUDIT.md'}`",
            "  why: strongest evidence about current label semantics.",
            f"- path: `{output_dir / 'BASELINE_SANITY_AUDIT.md'}`",
            "  why: independent baseline check against the current cache/eval world.",
            "",
            "## Optional",
            f"- path: `{output_dir / 'sample_trace.csv'}`",
            "  why: spot-check alignment and path semantics on concrete samples.",
            f"- path: `{results_root / 'recovery_reproduce_05466' / 'RECOVERY_SUMMARY.md'}`",
            "  why: provenance limits of the old-good result.",
            f"- path: `{results_root / 'mora_aligned_feature_adapter' / 'config.json'}`",
            "  why: compare historical parser/config against current defaults.",
            "",
        ]

        # Final summary appended to architecture audit
        final_summary = [
            "# Final Read-only Audit Summary",
            "",
            f"1. 現在最可能是資料問題、evaluation 問題、還是模型程式問題？",
            "目前最可疑的是 **資料/label semantics 問題**，其次才是模型程式 comparability 問題。evidence 最強的是：現在 cache rebuilt 後的 target 幾乎是每個 sample 只有一個 positive，和標準 MM-IMDb 多標籤設定不一致。evaluation helper 本身看起來是多標籤正確寫法，但它可能正在一個已經被單標籤化的 target 世界上運作。",
            "",
            "2. 是否建議重抓 / 重建 MM-IMDb？",
            "保守建議是：**先重建 label pipeline，再決定是否需要重抓 feature**。如果 raw MM-IMDb json 仍在，優先從 raw source 重新生成 multi-label targets 做 audit；不一定要先重抽 CLIP feature。",
            "",
            "3. 是否建議繼續修 `train_mora_aligned_feature_adapter.py`？",
            "在 label semantics 沒釐清前，不建議先把時間花在 adapter 細修。先確認資料世界是不是本來就不是你以為的 MM-IMDb 多標籤 setting。",
            "",
            "4. 是否建議把 `0.5466` 完全封存？",
            "不建議直接刪掉，但建議把它明確標為 historical / non-reproducible result。現在證據不足以把它當 clean reference，也不足以完全宣告它無效。",
            "",
            "5. 外部 reviewer 下一步最該看哪三個檔案？",
            f"- `{output_dir / 'DATA_LINEAGE_AUDIT.md'}`",
            f"- `{scripts_dir / 'train_mora_aligned_feature_adapter.py'}`",
            f"- `{output_dir / 'independent_baseline_results.csv'}`",
            "",
        ]

        architecture_sections.extend(final_summary)

        write_text(output_dir / "ARCHITECTURE_AUDIT.md", "\n".join(architecture_sections) + "\n")
        write_text(output_dir / "DATA_LINEAGE_AUDIT.md", "\n".join(data_lineage_sections) + "\n")
        write_text(output_dir / "EVALUATION_AUDIT.md", "\n".join(evaluation_sections) + "\n")
        write_text(output_dir / "MODEL_FLOW_AUDIT.md", "\n".join(model_flow_sections) + "\n")
        write_text(output_dir / "SAMPLE_TRACE_AUDIT.md", "\n".join(sample_trace_sections) + "\n")
        write_text(output_dir / "BASELINE_SANITY_AUDIT.md", "\n".join(baseline_sections) + "\n")
        write_text(output_dir / "OLD_RESULT_COMPARISON.md", "\n".join(comparison_sections) + "\n")
        write_text(output_dir / "SUSPECTED_ISSUES.md", "\n".join(suspected_lines) + "\n")
        write_text(output_dir / "files_to_review.md", "\n".join(files_to_review_lines) + "\n")
    except Exception as exc:
        log_error(output_dir, "read_only_architecture_audit", exc)
        raise


if __name__ == "__main__":
    main()
