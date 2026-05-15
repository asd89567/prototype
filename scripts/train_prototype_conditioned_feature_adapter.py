import argparse
import copy
import json
import random
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



# -----------------------------------------------------------------------------
# Self-contained utility functions
# These replace helper imports that may be missing in a partially restored repo.
# -----------------------------------------------------------------------------
def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path):
    return Path(path).expanduser().resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def apply_thresholds(prob, thresholds):
    prob = np.asarray(prob)
    if isinstance(thresholds, str):
        try:
            thresholds = json.loads(thresholds)
        except Exception:
            thresholds = float(thresholds)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim == 0:
        return (prob >= float(thresholds)).astype(np.float32)
    return (prob >= thresholds.reshape(1, -1)).astype(np.float32)


def mean_bce_loss(logits, y, eps: float = 1e-8) -> float:
    logits = np.asarray(logits, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    prob = sigmoid(logits)
    loss = -(
        y * np.log(np.clip(prob, eps, 1.0 - eps))
        + (1.0 - y) * np.log(np.clip(1.0 - prob, eps, 1.0 - eps))
    )
    return float(loss.mean())


def multilabel_metrics(logits, y, thresholds):
    logits = np.asarray(logits, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    pred = apply_thresholds(sigmoid(logits), thresholds)
    return {
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "sample_f1": float(f1_score(y, pred, average="samples", zero_division=0)),
        "accuracy": float((pred == y).all(axis=1).mean()),
        "bce_loss": mean_bce_loss(logits, y),
    }


def select_best_thresholds(logits, y, threshold_grid=None, threshold_strategies=None, tune_thresholds=True):
    if threshold_grid is None:
        threshold_grid = [0.5]
    if threshold_strategies is None:
        threshold_strategies = ["global", "per_class"]
    logits = np.asarray(logits, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    prob = sigmoid(logits)
    best = None

    def consider(mode, thresholds):
        nonlocal best
        metrics = multilabel_metrics(logits, y, thresholds)
        payload = {
            "threshold_mode": mode,
            "thresholds": thresholds.tolist() if hasattr(thresholds, "tolist") else thresholds,
            "metrics": metrics,
        }
        if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
            best = payload

    if not tune_thresholds:
        consider("fixed_0.5", 0.5)
        return best

    if "global" in threshold_strategies:
        for threshold in threshold_grid:
            consider("global", float(threshold))

    if "per_class" in threshold_strategies:
        n_labels = y.shape[1]
        thresholds = np.zeros(n_labels, dtype=np.float32)
        for label_idx in range(n_labels):
            best_t = 0.5
            best_f1 = -1.0
            for threshold in threshold_grid:
                pred_j = (prob[:, label_idx] >= float(threshold)).astype(np.float32)
                f1 = f1_score(y[:, label_idx], pred_j, average="binary", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(threshold)
            thresholds[label_idx] = best_t
        consider("per_class", thresholds)

    if best is None:
        consider("fixed_0.5", 0.5)
    return best



def normalize_sample_id(value):
    """
    Normalize MM-IMDb sample ids.

    Feature cache often stores ids like '0098333',
    while pandas may read metadata sample_id as int -> '98333'.
    This function pads numeric ids to 7 digits and removes optional tt prefix.
    """
    if value is None:
        return ""

    s = str(value).strip()

    if s.lower() in {"nan", "none", ""}:
        return ""

    # If path-like, use filename stem.
    s = Path(s).stem

    # Remove common IMDB prefix.
    if s.startswith("tt"):
        s = s[2:]

    # Handle pandas float-like ids, e.g. 98333.0
    if s.endswith(".0"):
        s = s[:-2]

    # Keep only digits if possible.
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return digits.zfill(7)

    return s


def load_metadata(metadata_csv):
    """
    Load metadata and preserve/normalize sample_id.

    Important:
    MM-IMDb ids must keep leading zeros, e.g. 0098333.
    """
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata_csv not found: {metadata_csv}")

    # Force sample_id as string so pandas does not drop leading zeros.
    df = pd.read_csv(metadata_csv, dtype={"sample_id": str, "id": str, "imdb_id": str})

    if "sample_id" not in df.columns:
        if "id" in df.columns:
            df["sample_id"] = df["id"]
        elif "imdb_id" in df.columns:
            df["sample_id"] = df["imdb_id"]

    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].map(normalize_sample_id)

    return df


def _load_npy_if_exists(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required feature file not found: {path}")
    return np.load(path, allow_pickle=True)


def load_split_arrays(feature_dir, split):
    feature_dir = Path(feature_dir)
    image = _load_npy_if_exists(feature_dir / f"{split}_image.npy")
    text = _load_npy_if_exists(feature_dir / f"{split}_text.npy")
    payload = {"image": image.astype(np.float32), "text": text.astype(np.float32)}
    for label_name in [f"{split}_label.npy", f"{split}_labels.npy", f"{split}_y.npy"]:
        p = feature_dir / label_name
        if p.exists():
            payload["label"] = np.load(p, allow_pickle=True)
            break
    for id_name in [f"{split}_sample_id.npy", f"{split}_sample_ids.npy", f"{split}_ids.npy"]:
        p = feature_dir / id_name
        if p.exists():
            ids = np.load(p, allow_pickle=True).astype(str)
            payload["sample_id"] = ids
            payload["sample_ids"] = ids
            break
    return payload


def ordered_split_metadata(metadata_df, feature_dir, split):
    """
    Return metadata rows ordered to match cached feature rows.

    Priority:
    1. If {split}_sample_ids.npy exists, align by normalized sample_id.
    2. Else if metadata has split column, use rows where split == split.
    3. Else assume metadata is already ordered and slice by split sizes.
    """
    feature_dir = Path(feature_dir)
    metadata_df = metadata_df.copy()

    if "sample_id" in metadata_df.columns:
        metadata_df["sample_id"] = metadata_df["sample_id"].map(normalize_sample_id)

    id_candidates = [
        feature_dir / f"{split}_sample_ids.npy",
        feature_dir / f"{split}_sample_id.npy",
        feature_dir / f"{split}_ids.npy",
    ]

    for id_path in id_candidates:
        if id_path.exists():
            raw_ids = np.load(id_path, allow_pickle=True).astype(str)
            sample_ids = [normalize_sample_id(x) for x in raw_ids]

            if "sample_id" not in metadata_df.columns:
                raise ValueError(
                    f"{id_path.name} exists, but metadata has no sample_id/id/imdb_id column."
                )

            # Drop duplicated normalized ids if any, keeping first.
            dedup = metadata_df.drop_duplicates("sample_id", keep="first")
            indexed = dedup.set_index("sample_id", drop=False)

            missing = [sid for sid in sample_ids if sid not in indexed.index]
            if missing:
                raise ValueError(
                    f"{len(missing)} sample ids from {id_path.name} not found in metadata after normalization. "
                    f"Examples: {missing[:5]}"
                )

            return indexed.loc[sample_ids].reset_index(drop=True)

    if "split" in metadata_df.columns:
        split_df = metadata_df[metadata_df["split"].astype(str).str.lower() == split.lower()].copy()
        if len(split_df) > 0:
            return split_df.reset_index(drop=True)

    image_path = feature_dir / f"{split}_image.npy"
    if not image_path.exists():
        raise FileNotFoundError(f"Cannot infer split metadata; missing {image_path}")

    n = len(np.load(image_path, mmap_mode="r", allow_pickle=True))

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


def load_mmimdb_payload(feature_dir: Path, metadata_csv: Path) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict[str, pd.DataFrame], List[str], Dict]:
    metadata_df = load_metadata(metadata_csv)
    feature_meta = load_feature_metadata(feature_dir)
    label_to_id = build_label_mapping(metadata_df, feature_meta)
    split_meta = {split: ordered_split_metadata(metadata_df, feature_dir, split) for split in ("train", "val", "test")}
    labels, class_names = build_multilabel_targets(metadata_df, split_meta, label_to_id)
    features = {split: load_split_arrays(feature_dir, split) for split in ("train", "val", "test")}
    return features, labels, split_meta, class_names, feature_meta


def concat_full(split_payload: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([split_payload["image"], split_payload["text"]], axis=1).astype(np.float32)


def positive_weight(y: np.ndarray, device: torch.device) -> Optional[torch.Tensor]:
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    weights = np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(*arrays: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    tensors = []
    for array in arrays:
        dtype = torch.long if array.dtype.kind in {"i", "u"} and array.ndim == 1 else torch.float32
        tensors.append(torch.tensor(array, dtype=dtype))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=generator)




def device_from_args(args):
    import torch
    if getattr(args, "device", "auto") == "cuda":
        return torch.device("cuda")
    if getattr(args, "device", "auto") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RepresentationClassifier(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(latent_dim, output_dim)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)

    def forward(self, x: torch.Tensor, return_z: bool = False):
        z = self.project(x)
        logits = self.classifier(z)
        if return_z:
            return logits, z
        return logits


class BottleneckAdapter(nn.Module):
    def __init__(self, latent_dim: int, rank: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, rank),
            nn.ReLU(),
            nn.Linear(rank, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class StaticFeatureAdapter(nn.Module):
    def __init__(
        self,
        base_model: RepresentationClassifier,
        latent_dim: int,
        adapter_rank: int,
        freeze_base: bool,
    ):
        super().__init__()
        self.projector = copy.deepcopy(base_model.projector)
        self.classifier = copy.deepcopy(base_model.classifier)
        self.adapter = BottleneckAdapter(latent_dim, adapter_rank)
        if freeze_base:
            for parameter in self.projector.parameters():
                parameter.requires_grad = False
            for parameter in self.classifier.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor, return_details: bool = False):
        z = self.projector(x)
        base_logits = self.classifier(z)
        delta = self.adapter(z)
        z_adapted = z + delta
        logits = self.classifier(z_adapted)
        if return_details:
            return logits, {
                "z_before": z,
                "z_after": z_adapted,
                "delta": delta,
                "base_logits": base_logits,
                "alpha": torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device),
            }
        return logits


class MixtureFeatureAdapter(nn.Module):
    def __init__(
        self,
        base_model: RepresentationClassifier,
        latent_dim: int,
        adapter_rank: int,
        output_dim: int,
        k: int,
        freeze_base: bool,
        router_hidden_dim: int,
        dropout: float,
        use_router: bool = True,
        enable_safe_beta: bool = False,
        beta_bias: float = 0.0,
        branch_router_layernorm: bool = False,
        branch_str: str = "",
    ):
        super().__init__()
        self.enable_safe_beta = enable_safe_beta
        self.branch_str = branch_str
        self.k = k
        self.use_router = use_router
        # Optional runtime override used by beta ablations.
        # None = learned beta; "zero" = beta=0; "one" = beta=1.
        self.beta_override_mode = None

        self.projector = copy.deepcopy(base_model.projector)
        self.classifier = copy.deepcopy(base_model.classifier)
        self.adapters = nn.ModuleList([BottleneckAdapter(latent_dim, adapter_rank) for _ in range(k)])

        self.branch_ln = nn.LayerNorm(latent_dim) if branch_router_layernorm else nn.Identity()
        router_input_dim = latent_dim + output_dim * 3
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, k),
        )

        if self.enable_safe_beta:
            self.static_adapter = BottleneckAdapter(latent_dim, adapter_rank)
            self.missing_type_emb = nn.Parameter(torch.zeros(latent_dim))
            # Keep this deliberately simple and consistent between train/eval:
            # beta sees branch-normalized z and a learned missing-type embedding.
            beta_input_dim = latent_dim * 2
            self.beta_mlp = nn.Sequential(
                nn.Linear(beta_input_dim, router_hidden_dim),
                nn.ReLU(),
                nn.Linear(router_hidden_dim, 1),
            )
            nn.init.constant_(self.beta_mlp[-1].bias, beta_bias)

        if freeze_base:
            for parameter in self.projector.parameters():
                parameter.requires_grad = False
            for parameter in self.classifier.parameters():
                parameter.requires_grad = False

    def router_features(self, z: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(base_logits)
        uncertainty = prob * (1.0 - prob)
        return torch.cat([z, base_logits, prob, uncertainty], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        oracle_cluster: Optional[torch.Tensor] = None,
        alpha_override: Optional[torch.Tensor] = None,
        beta_override: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        z = self.projector(x)
        base_logits = self.classifier(z)

        router_z = self.branch_ln(z)
        router_logits = self.router(self.router_features(router_z, base_logits))

        if alpha_override is not None:
            alpha = alpha_override.to(dtype=z.dtype, device=z.device)
        elif oracle_cluster is not None:
            alpha = torch.nn.functional.one_hot(oracle_cluster.long(), num_classes=self.k).to(dtype=z.dtype, device=z.device)
        else:
            alpha = torch.softmax(router_logits, dim=1)

        expert_deltas = torch.stack([adapter(z) for adapter in self.adapters], dim=1)
        mixture_delta = torch.sum(alpha.unsqueeze(-1) * expert_deltas, dim=1)

        if self.enable_safe_beta:
            static_delta = self.static_adapter(z)
            z_static = z + static_delta
            beta_input = torch.cat(
                [router_z, self.missing_type_emb.unsqueeze(0).expand(z.size(0), -1)],
                dim=1,
            )
            beta = torch.sigmoid(self.beta_mlp(beta_input))
            if beta_override is not None:
                beta = beta_override.to(dtype=z.dtype, device=z.device)
                if beta.ndim == 1:
                    beta = beta.unsqueeze(1)
            elif self.beta_override_mode == "zero":
                beta = torch.zeros((z.size(0), 1), dtype=z.dtype, device=z.device)
            elif self.beta_override_mode == "one":
                beta = torch.ones((z.size(0), 1), dtype=z.dtype, device=z.device)
            z_adapted = z_static + beta * mixture_delta
            delta = z_adapted - z
        else:
            static_delta = torch.zeros_like(z)
            z_static = z
            beta = torch.ones((z.size(0), 1), dtype=z.dtype, device=z.device)
            delta = mixture_delta
            z_adapted = z + delta

        logits = self.classifier(z_adapted)
        if return_details:
            return logits, {
                "z_before": z,
                "z_static": z_static,
                "z_after": z_adapted,
                "static_delta": static_delta,
                "mixture_delta": mixture_delta,
                "delta": delta,
                "beta": beta,
                "base_logits": base_logits,
                "router_logits": router_logits,
                "alpha": alpha,
            }
        return logits


@torch.inference_mode()
def predict_representation(model: RepresentationClassifier, x: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_out, z_out = [], []
    for start in range(0, len(x), 512):
        batch = torch.tensor(x[start : start + 512], dtype=torch.float32, device=device)
        logits, z = model(batch, return_z=True)
        logits_out.append(logits.cpu().numpy())
        z_out.append(z.cpu().numpy())
    return np.concatenate(logits_out, axis=0), np.concatenate(z_out, axis=0)


@torch.inference_mode()
def predict_adapter(model: nn.Module, x: np.ndarray, device: torch.device, clusters: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
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
        batch = torch.tensor(x[start : start + 512], dtype=torch.float32, device=device)
        cluster_tensor = None
        if clusters is not None:
            cluster_tensor = torch.tensor(clusters[start : start + 512], dtype=torch.long, device=device)
        if isinstance(model, MixtureFeatureAdapter):
            logits, details = model(batch, oracle_cluster=cluster_tensor, return_details=True)
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


def train_representation_classifier(
    model: RepresentationClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    stage: str,
    training_rows: List[Dict],
) -> Tuple[RepresentationClassifier, Dict]:
    model.to(device)
    pos_weight = None if args.no_pos_weight else positive_weight(y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(x_train.astype(np.float32), y_train.astype(np.float32), batch_size=args.batch_size, shuffle=True, seed=args.seed)
    best_state = None
    best_threshold_info = None
    best_score = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_logits, _ = predict_representation(model, x_val, device)
        threshold_info = select_best_thresholds(val_logits, y_val, args.threshold_grid, args.threshold_strategies, args.tune_thresholds)
        training_rows.append(
            {
                "stage": stage,
                "method": stage,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_macro_f1": threshold_info["metrics"]["macro_f1"],
                "val_micro_f1": threshold_info["metrics"]["micro_f1"],
                "val_sample_f1": threshold_info["metrics"]["sample_f1"],
                "val_bce_loss": threshold_info["metrics"]["bce_loss"],
                "threshold_strategy": threshold_info["threshold_mode"],
            }
        )
        if threshold_info["metrics"]["macro_f1"] > best_score:
            best_score = float(threshold_info["metrics"]["macro_f1"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_threshold_info = threshold_info
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_threshold_info


def entropy_loss(alpha: torch.Tensor) -> torch.Tensor:
    entropy = -(alpha * torch.log(torch.clamp(alpha, min=1e-8))).sum(dim=1).mean()
    return entropy


def train_adapter_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    z_full_train: np.ndarray,
    cluster_train: Optional[np.ndarray],
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    stage: str,
    method: str,
    lambda_align: float,
    lambda_router: float,
    use_oracle_clusters: bool,
    val_oracle_clusters: Optional[np.ndarray],
    training_rows: List[Dict],
) -> Tuple[nn.Module, Dict]:
    model.to(device)
    pos_weight = None if args.no_pos_weight else positive_weight(y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    router_criterion = nn.CrossEntropyLoss()
    align_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    arrays = [x_train.astype(np.float32), y_train.astype(np.float32), z_full_train.astype(np.float32)]
    if cluster_train is not None:
        arrays.append(cluster_train.astype(np.int64))
    loader = make_loader(*arrays, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    best_state = None
    best_threshold_info = None
    best_score = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch_x = batch[0].to(device)
            batch_y = batch[1].to(device)
            batch_z_full = batch[2].to(device)
            batch_cluster = batch[3].to(device) if len(batch) > 3 else None
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, MixtureFeatureAdapter):
                oracle_for_batch = batch_cluster if use_oracle_clusters else None
                logits, details = model(batch_x, oracle_cluster=oracle_for_batch, return_details=True)
            else:
                logits, details = model(batch_x, return_details=True)
            loss = criterion(logits, batch_y)
            if lambda_align > 0:
                loss = loss + lambda_align * align_criterion(details["z_after"], batch_z_full)
            if lambda_router > 0 and isinstance(model, MixtureFeatureAdapter) and batch_cluster is not None and not use_oracle_clusters:
                loss = loss + lambda_router * router_criterion(details["router_logits"], batch_cluster)
            if args.lambda_entropy > 0 and isinstance(model, MixtureFeatureAdapter) and not use_oracle_clusters:
                loss = loss + args.lambda_entropy * entropy_loss(details["alpha"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if use_oracle_clusters:
            val_details = predict_adapter(model, x_val, device, val_oracle_clusters)
        else:
            val_details = predict_adapter(model, x_val, device)
        threshold_info = select_best_thresholds(val_details["logits"], y_val, args.threshold_grid, args.threshold_strategies, args.tune_thresholds)
        training_rows.append(
            {
                "stage": stage,
                "method": method,
                "epoch": epoch,
                "K": getattr(model, "k", 1),
                "lambda_align": lambda_align,
                "lambda_router": lambda_router,
                "train_loss": float(np.mean(losses)),
                "val_macro_f1": threshold_info["metrics"]["macro_f1"],
                "val_micro_f1": threshold_info["metrics"]["micro_f1"],
                "val_sample_f1": threshold_info["metrics"]["sample_f1"],
                "val_bce_loss": threshold_info["metrics"]["bce_loss"],
                "threshold_strategy": threshold_info["threshold_mode"],
            }
        )
        if threshold_info["metrics"]["macro_f1"] > best_score:
            best_score = float(threshold_info["metrics"]["macro_f1"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_threshold_info = threshold_info
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_threshold_info


def evaluate_result_row(
    args: argparse.Namespace,
    method: str,
    logits: np.ndarray,
    y: np.ndarray,
    threshold_info: Dict,
    missing_macro: Optional[float],
    static_macro: Optional[float],
    k: Optional[int],
    lambda_align: float,
    lambda_router: float,
    deployable: bool,
) -> Dict:
    metrics = multilabel_metrics(logits, y, threshold_info["thresholds"])
    return {
        "dataset": args.dataset,
        "missing_type": args.missing_type,
        "method": method,
        "K": k if k is not None else "",
        "adapter_rank": args.adapter_rank,
        "lambda_align": lambda_align,
        "lambda_router": lambda_router,
        "freeze_base": bool(args.freeze_base),
        "threshold_strategy": threshold_info["threshold_mode"],
        "threshold": json.dumps(threshold_info["thresholds"]),
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "sample_f1": metrics["sample_f1"],
        "accuracy": metrics["accuracy"],
        "bce_loss": metrics["bce_loss"],
        "mean_predicted_positive_labels": float(apply_thresholds(sigmoid(logits), threshold_info["thresholds"]).sum(axis=1).mean()),
        "mean_true_positive_labels": float(y.sum(axis=1).mean()),
        "delta_vs_missing_only_macro_f1": float(metrics["macro_f1"] - missing_macro) if missing_macro is not None else float("nan"),
        "delta_vs_static_adapter_macro_f1": float(metrics["macro_f1"] - static_macro) if static_macro is not None else float("nan"),
        "deployable": bool(deployable),
    }


def per_label_rows(args: argparse.Namespace, method: str, logits: np.ndarray, y: np.ndarray, thresholds, class_names: List[str], k, lambda_align, lambda_router) -> List[Dict]:
    pred = apply_thresholds(sigmoid(logits), thresholds)
    rows = []
    for idx, label_name in enumerate(class_names):
        rows.append(
            {
                "dataset": args.dataset,
                "missing_type": args.missing_type,
                "method": method,
                "K": k if k is not None else "",
                "lambda_align": lambda_align,
                "lambda_router": lambda_router,
                "label_id": idx,
                "label_name": label_name,
                "label_f1": float(f1_score(y[:, idx], pred[:, idx], average="binary", zero_division=0)),
                "label_support": int(y[:, idx].sum()),
                "label_positive_rate": float(y[:, idx].mean()),
                "pred_positive_rate": float(pred[:, idx].mean()),
            }
        )
    return rows


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def fit_residual_clusters(residual_by_split: Dict[str, np.ndarray], k_values: List[int], seed: int) -> Dict[int, Dict]:
    out = {}
    train_normed = normalize_rows(residual_by_split["train"])
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=seed)
        train_labels = model.fit_predict(train_normed)
        centers = model.cluster_centers_
        split_labels = {"train": train_labels}
        for split in ("val", "test"):
            sims = normalize_rows(residual_by_split[split]) @ normalize_rows(centers).T
            split_labels[split] = np.argmax(sims, axis=1).astype(np.int64)
        out[k] = {"model": model, "centers": centers, "labels": split_labels}
    return out


def residual_cluster_summary_rows(cluster_payload: Dict[int, Dict], residual_by_split: Dict[str, np.ndarray], labels_by_split: Dict[str, np.ndarray], class_names: List[str]) -> List[Dict]:
    rows = []
    for k, payload in cluster_payload.items():
        for cluster_id in range(k):
            row = {"K": k, "cluster_id": cluster_id}
            for split in ("train", "val", "test"):
                mask = payload["labels"][split] == cluster_id
                row[f"n_{split}"] = int(mask.sum())
                if split == "train" and mask.any():
                    norms = np.linalg.norm(residual_by_split[split][mask], axis=1)
                    row["mean_residual_norm"] = float(norms.mean())
                    row["std_residual_norm"] = float(norms.std())
                    pos_rates = labels_by_split[split][mask].mean(axis=0)
                    top_idx = np.argsort(pos_rates)[::-1][:3]
                    row["top_labels_positive_rate"] = "; ".join(f"{class_names[idx]}:{pos_rates[idx]:.3f}" for idx in top_idx)
            row["cluster_center_norm"] = float(np.linalg.norm(payload["centers"][cluster_id]))
            rows.append(row)
    return rows


def cosine_mean(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    numerator = (a * b).sum(axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean(numerator / np.maximum(denom, eps)))


def alignment_row(method: str, split: str, z_full: np.ndarray, z_before: np.ndarray, z_after: np.ndarray, k, lambda_align, lambda_router) -> Dict:
    mse_before = float(np.mean((z_before - z_full) ** 2))
    mse_after = float(np.mean((z_after - z_full) ** 2))
    cosine_before = cosine_mean(z_before, z_full)
    cosine_after = cosine_mean(z_after, z_full)
    return {
        "split": split,
        "method": method,
        "K": k if k is not None else "",
        "lambda_align": lambda_align,
        "lambda_router": lambda_router,
        "mean_mse_to_full_z_before": mse_before,
        "mean_mse_to_full_z_after": mse_after,
        "mean_cosine_to_full_z_before": cosine_before,
        "mean_cosine_to_full_z_after": cosine_after,
        "delta_mse": mse_before - mse_after,
        "delta_cosine": cosine_after - cosine_before,
    }


def router_stats_rows(method: str, k: int, details_by_split: Dict[str, Dict[str, np.ndarray]], cluster_labels: Dict[str, np.ndarray], lambda_align: float, lambda_router: float) -> List[Dict]:
    rows = []
    for split, details in details_by_split.items():
        if "alpha" not in details:
            continue
        alpha = details["alpha"]
        if alpha.shape[1] != k:
            continue
        pred = alpha.argmax(axis=1)
        true = cluster_labels[split]
        entropy = -(alpha * np.log(np.clip(alpha, 1e-8, 1.0))).sum(axis=1)
        rows.append(
            {
                "split": split,
                "method": method,
                "K": k,
                "lambda_align": lambda_align,
                "lambda_router": lambda_router,
                "router_accuracy": float((pred == true).mean()),
                "router_macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
                "router_entropy_mean": float(entropy.mean()),
                "alpha_max_mean": float(alpha.max(axis=1).mean()),
                "alpha_max_std": float(alpha.max(axis=1).std()),
                "cluster_balance_predicted": json.dumps(np.bincount(pred, minlength=k).astype(int).tolist()),
                "cluster_balance_true": json.dumps(np.bincount(true, minlength=k).astype(int).tolist()),
            }
        )
    return rows


def load_harm_by_split(path: Path, split_meta: Dict[str, pd.DataFrame], missing_type: str, output_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype={"sample_id": str})
        column = f"{missing_type}_raw_harm"
        if column not in df.columns:
            return None
        out = {}
        for split, meta in split_meta.items():
            indexed = df[df["split"] == split].set_index("sample_id")
            sample_ids = meta["sample_id"].astype(str).tolist()
            missing = pd.Index(sample_ids).difference(indexed.index)
            if not missing.empty:
                raise ValueError(f"Harm file missing {split} ids: {missing.tolist()[:5]}")
            out[split] = indexed.loc[sample_ids, column].to_numpy(dtype=np.float32)
        return out
    except Exception as exc:
        log_error(output_dir, "harm_loading", exc)
        return None


def harm_group_masks(values: np.ndarray) -> Dict[str, np.ndarray]:
    high_cut = np.quantile(values, 0.70)
    low_cut = np.quantile(values, 0.30)
    return {
        "high_harm_top30": values >= high_cut,
        "middle_harm": (values < high_cut) & (values > low_cut),
        "low_harm_bottom30": values <= low_cut,
        "negative_harm": values <= 0.0,
    }


def group_metric_rows(
    args: argparse.Namespace,
    method: str,
    logits: np.ndarray,
    y: np.ndarray,
    thresholds,
    z_full: np.ndarray,
    z_before: np.ndarray,
    z_after: np.ndarray,
    delta: np.ndarray,
    harm_values: Optional[np.ndarray],
    residual_norm: np.ndarray,
    k,
    lambda_align,
    lambda_router,
) -> List[Dict]:
    if harm_values is None:
        return []
    rows = []
    before_mse = ((z_before - z_full) ** 2).mean(axis=1)
    after_mse = ((z_after - z_full) ** 2).mean(axis=1)
    for group, mask in harm_group_masks(harm_values).items():
        if not mask.any():
            continue
        metrics = multilabel_metrics(logits[mask], y[mask], thresholds)
        rows.append(
            {
                "dataset": args.dataset,
                "missing_type": args.missing_type,
                "method": method,
                "K": k if k is not None else "",
                "lambda_align": lambda_align,
                "lambda_router": lambda_router,
                "group": group,
                "n_samples": int(mask.sum()),
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "sample_f1": metrics["sample_f1"],
                "bce_loss": metrics["bce_loss"],
                "mean_residual_norm": float(residual_norm[mask].mean()),
                "mean_delta_norm": float(np.linalg.norm(delta[mask], axis=1).mean()),
                "mean_alignment_improvement": float((before_mse[mask] - after_mse[mask]).mean()),
            }
        )
    return rows


def evaluate_stage_model(
    args: argparse.Namespace,
    model: RepresentationClassifier,
    x_by_split: Dict[str, np.ndarray],
    y_by_split: Dict[str, np.ndarray],
    threshold_info: Dict,
    device: torch.device,
    method: str,
) -> pd.DataFrame:
    rows = []
    for split in ("val", "test"):
        logits, _ = predict_representation(model, x_by_split[split], device)
        metrics = multilabel_metrics(logits, y_by_split[split], threshold_info["thresholds"])
        rows.append(
            {
                "dataset": args.dataset,
                "missing_type": args.missing_type,
                "method": method,
                "split": split,
                "threshold_strategy": threshold_info["threshold_mode"],
                "threshold": json.dumps(threshold_info["thresholds"]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def make_summary(output_dir: Path) -> None:
    def read_csv_or_empty(path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    results_path = output_dir / "mmimdb_results.csv"
    full_path = output_dir / "full_teacher_results.csv"
    missing_path = output_dir / "missing_base_results.csv"
    cluster_path = output_dir / "residual_cluster_summary.csv"
    router_path = output_dir / "router_prediction_stats.csv"
    align_path = output_dir / "residual_alignment_stats.csv"
    if not results_path.exists():
        return
    results = read_csv_or_empty(results_path)
    full = read_csv_or_empty(full_path)
    missing = read_csv_or_empty(missing_path)
    clusters = read_csv_or_empty(cluster_path)
    routers = read_csv_or_empty(router_path)
    align = read_csv_or_empty(align_path)
    if results.empty:
        return

    test_full = full[full["split"] == "test"] if not full.empty else pd.DataFrame()
    test_missing = missing[missing["split"] == "test"] if not missing.empty else pd.DataFrame()
    best = results.sort_values("macro_f1", ascending=False).head(1)
    best_proto = results[results["method"].eq("prototype_conditioned_feature_adapter")].sort_values("macro_f1", ascending=False).head(1)
    best_static = results[results["method"].eq("static_feature_adapter")].sort_values("macro_f1", ascending=False).head(1)
    best_oracle = results[results["method"].eq("oracle_cluster_adapter_upper_bound")].sort_values("macro_f1", ascending=False).head(1)
    best_mixture = results[results["method"].eq("mixture_feature_adapter_unsupervised")].sort_values("macro_f1", ascending=False).head(1)
    missing_row = results[results["method"].eq("missing_only")].head(1)

    lines = [
        "# Prototype-Conditioned Feature Adapter RUN SUMMARY",
        "",
        "## 1. 本輪目標",
        "這輪是 representation-level prototype-conditioned residual adapter diagnostic，不是 output-head gate，不是 MoRA，也不是 CLIP fine-tuning。所有模型都只使用 cached frozen CLIP features。",
        "",
        "## 2. 為什麼改方向",
        "前面 output-wise gate 雖然有 oracle gap，但可部署 gate 沒有穩定縮小 gap，而且方法逐漸變成 prediction head selection。這輪把補償移到 representation 層，檢查 sample-aware residual direction 是否比 static adapter 更有研究空間。",
        "",
        "## 3. 方法說明",
        "full teacher: image + text -> z_full",
        "",
        "missing base: text -> z_missing",
        "",
        "residual: r = z_full - z_missing",
        "",
        "cluster residuals -> compensation prototypes",
        "",
        "router(text-only info) -> adapter mixture weights",
        "",
        "z_adapted = z_missing + sum alpha_k Adapter_k(z_missing)",
        "",
        "## 4. Baseline 結果",
    ]
    if not test_full.empty:
        lines.append(f"- full teacher test macro-F1: {test_full.iloc[0]['macro_f1']:.4f}")
    if not missing_row.empty:
        lines.append(f"- missing_only test macro-F1: {missing_row.iloc[0]['macro_f1']:.4f}")
    if not best_static.empty:
        lines.append(f"- best static_feature_adapter test macro-F1: {best_static.iloc[0]['macro_f1']:.4f}")
    lines.extend(["", "## 5. Prototype-conditioned 結果"])
    if not best_mixture.empty:
        lines.append(f"- best mixture_feature_adapter_unsupervised: {best_mixture.iloc[0]['macro_f1']:.4f}")
    if not best_proto.empty:
        row = best_proto.iloc[0]
        lines.append(f"- best prototype_conditioned_feature_adapter: {row['macro_f1']:.4f} (K={row['K']}, lambda_align={row['lambda_align']}, lambda_router={row['lambda_router']})")
    if not best_oracle.empty:
        row = best_oracle.iloc[0]
        lines.append(f"- best oracle_cluster_adapter_upper_bound: {row['macro_f1']:.4f} (K={row['K']}, lambda_align={row['lambda_align']})")
    lines.extend(["", "## 6. 是否超過 static adapter"])
    if not best_proto.empty and not best_static.empty and not missing_row.empty:
        proto = best_proto.iloc[0]
        static = best_static.iloc[0]
        missing_macro = missing_row.iloc[0]["macro_f1"]
        lines.append(f"- best prototype-conditioned macro-F1: {proto['macro_f1']:.4f}")
        lines.append(f"- delta vs missing_only: {proto['macro_f1'] - missing_macro:+.4f}")
        lines.append(f"- delta vs static_feature_adapter: {proto['macro_f1'] - static['macro_f1']:+.4f}")
    if not best_oracle.empty and not best_static.empty:
        lines.append(f"- oracle cluster upper bound delta vs static: {best_oracle.iloc[0]['macro_f1'] - best_static.iloc[0]['macro_f1']:+.4f}")
    lines.extend(["", "## 7. Router 是否學到 residual prototype"])
    if not routers.empty:
        proto_routers = routers[(routers["method"] == "prototype_conditioned_feature_adapter") & (routers["split"] == "test")]
        if not proto_routers.empty:
            row = proto_routers.sort_values("router_accuracy", ascending=False).iloc[0]
            lines.append(f"- best test router accuracy: {row['router_accuracy']:.4f}")
            lines.append(f"- best test router macro-F1: {row['router_macro_f1']:.4f}")
            lines.append(f"- alpha entropy mean: {row['router_entropy_mean']:.4f}")
            lines.append(f"- alpha max mean: {row['alpha_max_mean']:.4f}")
    lines.extend(["", "## 8. Representation 是否靠近 full"])
    if not align.empty:
        test_align = align[align["split"] == "test"].sort_values("delta_mse", ascending=False)
        if not test_align.empty:
            row = test_align.iloc[0]
            lines.append(f"- largest test MSE improvement method: {row['method']}")
            lines.append(f"- MSE before / after: {row['mean_mse_to_full_z_before']:.6f} / {row['mean_mse_to_full_z_after']:.6f}")
            lines.append(f"- cosine before / after: {row['mean_cosine_to_full_z_before']:.4f} / {row['mean_cosine_to_full_z_after']:.4f}")
    lines.extend(["", "## 9. 對 thesis 的意義"])
    interpretation = "結果不足，請查看 mmimdb_results.csv。"
    if not best_proto.empty and not best_static.empty and not best_oracle.empty:
        proto_macro = float(best_proto.iloc[0]["macro_f1"])
        static_macro = float(best_static.iloc[0]["macro_f1"])
        oracle_macro = float(best_oracle.iloc[0]["macro_f1"])
        if proto_macro > static_macro:
            interpretation = "prototype-conditioned adapter 超過 static adapter，representation-level sample-aware compensation 有初步證據，可以再往更正式的 adapter/MoRA scaling 方向探索。"
        elif oracle_macro > static_macro + 0.01:
            interpretation = "prototype-conditioned adapter 接近或低於 static adapter，但 oracle cluster 明顯更高，代表 residual prototype 有空間，主要問題是 router 學不好。"
        else:
            interpretation = "static adapter 已經不差，oracle cluster 也沒有明顯拉開；目前 sample-aware prototype routing 的必要性有限。"
    lines.append(interpretation)
    lines.extend(
        [
            "",
            "## 10. 下一步建議",
            "- 先不要把 output-wise gate 當主方法；它比較適合作為 diagnostic。",
            "- 是否接 MoRA 要看本輪 prototype-conditioned adapter 是否穩定超過 static adapter；若沒有，先擴大 MM-IMDb subset 或跑 text_missing。",
            "- Food101 目前只有 template_text，不適合作為正式 missing text-image 證據；第二資料集建議優先找有真文字的 Hateful Memes 或回到 MM-IMDb full subset。",
        ]
    )
    output_dir.joinpath("RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    feature_dir = resolve_path(args.feature_dir)
    metadata_csv = resolve_path(args.metadata_csv)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    harm_target_csv = resolve_path(args.harm_target_csv)
    (output_dir / "errors.log").write_text("", encoding="utf-8")
    device = device_from_args(args)
    config_payload = vars(args).copy()
    config_payload["device_resolved"] = str(device)
    save_json(config_payload, output_dir / "config.json")

    training_rows: List[Dict] = []
    result_rows: List[Dict] = []
    per_label_all: List[Dict] = []
    cluster_summary_all: List[Dict] = []
    router_stats_all: List[Dict] = []
    alignment_rows_all: List[Dict] = []
    group_rows_all: List[Dict] = []

    try:
        features, labels, split_meta, class_names, feature_meta = load_mmimdb_payload(feature_dir, metadata_csv)
        audit = {
            "feature_dir": str(feature_dir),
            "metadata_csv": str(metadata_csv),
            "feature_metadata": feature_meta,
            "class_names": class_names,
            "splits": {
                split: {
                    "image_shape": list(features[split]["image"].shape),
                    "text_shape": list(features[split]["text"].shape),
                    "label_shape": list(labels[split].shape),
                    "n_sample_ids": int(len(features[split]["sample_ids"])),
                }
                for split in ("train", "val", "test")
            },
        }
        save_json(audit, output_dir / "audit.json")
    except Exception as exc:
        log_error(output_dir, "audit_or_loading", exc)
        raise

    text_by_split = {split: features[split]["text"].astype(np.float32) for split in ("train", "val", "test")}
    full_by_split = {split: concat_full(features[split]) for split in ("train", "val", "test")}
    output_dim = labels["train"].shape[1]
    full_input_dim = full_by_split["train"].shape[1]
    text_dim = text_by_split["train"].shape[1]

    full_teacher = RepresentationClassifier(full_input_dim, args.latent_dim, args.hidden_dim, output_dim, args.dropout)
    full_teacher, full_threshold_info = train_representation_classifier(
        full_teacher,
        full_by_split["train"],
        labels["train"],
        full_by_split["val"],
        labels["val"],
        args,
        device,
        "full_teacher",
        training_rows,
    )
    torch.save(full_teacher.state_dict(), output_dir / "full_teacher.pt")
    evaluate_stage_model(args, full_teacher, full_by_split, labels, full_threshold_info, device, "full_teacher").to_csv(output_dir / "full_teacher_results.csv", index=False)

    missing_base = RepresentationClassifier(text_dim, args.latent_dim, args.hidden_dim, output_dim, args.dropout)
    missing_base, missing_threshold_info = train_representation_classifier(
        missing_base,
        text_by_split["train"],
        labels["train"],
        text_by_split["val"],
        labels["val"],
        args,
        device,
        "missing_base",
        training_rows,
    )
    torch.save(missing_base.state_dict(), output_dir / "missing_base.pt")
    evaluate_stage_model(args, missing_base, text_by_split, labels, missing_threshold_info, device, "missing_only").to_csv(output_dir / "missing_base_results.csv", index=False)

    full_logits_by_split, z_full_by_split = {}, {}
    missing_logits_by_split, z_missing_by_split = {}, {}
    for split in ("train", "val", "test"):
        full_logits_by_split[split], z_full_by_split[split] = predict_representation(full_teacher, full_by_split[split], device)
        missing_logits_by_split[split], z_missing_by_split[split] = predict_representation(missing_base, text_by_split[split], device)

    residual_by_split = {split: z_full_by_split[split] - z_missing_by_split[split] for split in ("train", "val", "test")}
    residual_norm_by_split = {split: np.linalg.norm(residual_by_split[split], axis=1) for split in ("train", "val", "test")}
    cluster_payload = fit_residual_clusters(residual_by_split, args.cluster_k_list, args.seed)
    cluster_summary_all.extend(residual_cluster_summary_rows(cluster_payload, residual_by_split, labels, class_names))

    harm_by_split = load_harm_by_split(harm_target_csv, split_meta, args.missing_type, output_dir)

    missing_test_logits = missing_logits_by_split["test"]
    missing_test_metrics = multilabel_metrics(missing_test_logits, labels["test"], missing_threshold_info["thresholds"])
    missing_macro = missing_test_metrics["macro_f1"]
    static_best_macro: Optional[float] = None

    result_rows.append(
        evaluate_result_row(args, "missing_only", missing_test_logits, labels["test"], missing_threshold_info, None, None, None, 0.0, 0.0, True)
    )
    per_label_all.extend(per_label_rows(args, "missing_only", missing_test_logits, labels["test"], missing_threshold_info["thresholds"], class_names, None, 0.0, 0.0))
    z_before = z_missing_by_split["test"]
    z_after = z_missing_by_split["test"]
    delta = np.zeros_like(z_before)
    alignment_rows_all.append(alignment_row("missing_only", "test", z_full_by_split["test"], z_before, z_after, None, 0.0, 0.0))
    group_rows_all.extend(
        group_metric_rows(
            args,
            "missing_only",
            missing_test_logits,
            labels["test"],
            missing_threshold_info["thresholds"],
            z_full_by_split["test"],
            z_before,
            z_after,
            delta,
            harm_by_split["test"] if harm_by_split else None,
            residual_norm_by_split["test"],
            None,
            0.0,
            0.0,
        )
    )

    static_rows_cache: List[Tuple[float, Dict, Dict, StaticFeatureAdapter]] = []
    if "static_feature_adapter" in args.methods:
        for lambda_align in args.lambda_align_list:
            try:
                model = StaticFeatureAdapter(missing_base, args.latent_dim, args.adapter_rank, args.freeze_base)
                model, threshold_info = train_adapter_model(
                    model,
                    text_by_split["train"],
                    labels["train"],
                    z_full_by_split["train"],
                    None,
                    text_by_split["val"],
                    labels["val"],
                    args,
                    device,
                    "adapter",
                    "static_feature_adapter",
                    lambda_align,
                    0.0,
                    False,
                    None,
                    training_rows,
                )
                details_by_split = {split: predict_adapter(model, text_by_split[split], device) for split in ("val", "test")}
                test_details = details_by_split["test"]
                row = evaluate_result_row(args, "static_feature_adapter", test_details["logits"], labels["test"], threshold_info, missing_macro, static_best_macro, None, lambda_align, 0.0, True)
                result_rows.append(row)
                static_rows_cache.append((row["macro_f1"], threshold_info, test_details, model))
                per_label_all.extend(per_label_rows(args, "static_feature_adapter", test_details["logits"], labels["test"], threshold_info["thresholds"], class_names, None, lambda_align, 0.0))
                alignment_rows_all.append(alignment_row("static_feature_adapter", "test", z_full_by_split["test"], test_details["z_before"], test_details["z_after"], None, lambda_align, 0.0))
                group_rows_all.extend(
                    group_metric_rows(
                        args,
                        "static_feature_adapter",
                        test_details["logits"],
                        labels["test"],
                        threshold_info["thresholds"],
                        z_full_by_split["test"],
                        test_details["z_before"],
                        test_details["z_after"],
                        test_details["delta"],
                        harm_by_split["test"] if harm_by_split else None,
                        residual_norm_by_split["test"],
                        None,
                        lambda_align,
                        0.0,
                    )
                )
                torch.save(model.state_dict(), output_dir / f"static_feature_adapter_align{lambda_align}.pt")
            except Exception as exc:
                log_error(output_dir, f"static_feature_adapter_align{lambda_align}", exc)
        if static_rows_cache:
            static_best_macro = max(item[0] for item in static_rows_cache)
            for row in result_rows:
                if row["method"] == "static_feature_adapter":
                    row["delta_vs_static_adapter_macro_f1"] = row["macro_f1"] - static_best_macro
            result_rows[0]["delta_vs_static_adapter_macro_f1"] = missing_macro - static_best_macro

    for method in ("mixture_feature_adapter_unsupervised", "prototype_conditioned_feature_adapter", "oracle_cluster_adapter_upper_bound"):
        if method not in args.methods:
            continue
        for k in args.cluster_k_list:
            lambda_router_values = [0.0] if method != "prototype_conditioned_feature_adapter" else args.lambda_router_list
            for lambda_align in args.lambda_align_list:
                for lambda_router in lambda_router_values:
                    if method == "prototype_conditioned_feature_adapter" and lambda_router == 0.0 and 0.1 in args.lambda_router_list:
                        # Keep the unsupervised mixture as the explicit no-router-supervision baseline.
                        continue
                    try:
                        use_oracle = method == "oracle_cluster_adapter_upper_bound"
                        model = MixtureFeatureAdapter(
                            missing_base,
                            args.latent_dim,
                            args.adapter_rank,
                            output_dim,
                            k,
                            args.freeze_base,
                            args.hidden_dim,
                            args.dropout,
                        )
                        cluster_labels = cluster_payload[k]["labels"]
                        train_clusters = cluster_labels["train"] if method in {"prototype_conditioned_feature_adapter", "oracle_cluster_adapter_upper_bound"} else None
                        val_clusters = cluster_labels["val"] if use_oracle else None
                        model, threshold_info = train_adapter_model(
                            model,
                            text_by_split["train"],
                            labels["train"],
                            z_full_by_split["train"],
                            train_clusters,
                            text_by_split["val"],
                            labels["val"],
                            args,
                            device,
                            "adapter",
                            method,
                            lambda_align,
                            lambda_router,
                            use_oracle,
                            val_clusters,
                            training_rows,
                        )
                        details_by_split = {}
                        for split in ("train", "val", "test"):
                            oracle_clusters = cluster_labels[split] if use_oracle else None
                            details_by_split[split] = predict_adapter(model, text_by_split[split], device, oracle_clusters)
                        test_details = details_by_split["test"]
                        row = evaluate_result_row(
                            args,
                            method,
                            test_details["logits"],
                            labels["test"],
                            threshold_info,
                            missing_macro,
                            static_best_macro,
                            k,
                            lambda_align,
                            lambda_router,
                            not use_oracle,
                        )
                        result_rows.append(row)
                        per_label_all.extend(per_label_rows(args, method, test_details["logits"], labels["test"], threshold_info["thresholds"], class_names, k, lambda_align, lambda_router))
                        alignment_rows_all.append(alignment_row(method, "test", z_full_by_split["test"], test_details["z_before"], test_details["z_after"], k, lambda_align, lambda_router))
                        router_stats_all.extend(router_stats_rows(method, k, details_by_split, cluster_labels, lambda_align, lambda_router))
                        group_rows_all.extend(
                            group_metric_rows(
                                args,
                                method,
                                test_details["logits"],
                                labels["test"],
                                threshold_info["thresholds"],
                                z_full_by_split["test"],
                                test_details["z_before"],
                                test_details["z_after"],
                                test_details["delta"],
                                harm_by_split["test"] if harm_by_split else None,
                                residual_norm_by_split["test"],
                                k,
                                lambda_align,
                                lambda_router,
                            )
                        )
                        torch.save(model.state_dict(), output_dir / f"{method}_K{k}_align{lambda_align}_router{lambda_router}.pt")
                    except Exception as exc:
                        log_error(output_dir, f"{method}_K{k}_align{lambda_align}_router{lambda_router}", exc)

    results_df = pd.DataFrame(result_rows)
    if static_best_macro is not None:
        results_df["delta_vs_static_adapter_macro_f1"] = results_df["macro_f1"] - static_best_macro
    results_df = results_df.sort_values("macro_f1", ascending=False)
    results_df.to_csv(output_dir / "mmimdb_results.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output_dir / "training_log.csv", index=False)
    pd.DataFrame(cluster_summary_all).to_csv(output_dir / "residual_cluster_summary.csv", index=False)
    pd.DataFrame(router_stats_all).to_csv(output_dir / "router_prediction_stats.csv", index=False)
    pd.DataFrame(alignment_rows_all).to_csv(output_dir / "residual_alignment_stats.csv", index=False)
    pd.DataFrame(per_label_all).to_csv(output_dir / "per_label_f1.csv", index=False)
    pd.DataFrame(group_rows_all).to_csv(output_dir / "high_low_harm_group_metrics.csv", index=False)
    make_summary(output_dir)


if __name__ == "__main__":
    main()



# ===== Compatibility helpers for train_mora_aligned_feature_adapter.py =====

def device_from_args(args):
    import torch
    if getattr(args, "device", "auto") == "cuda":
        return torch.device("cuda")
    if getattr(args, "device", "auto") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(payload, path):
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def concat_full(split_payload):
    import numpy as np
    return np.concatenate(
        [split_payload["image"], split_payload["text"]],
        axis=1
    ).astype("float32")


def positive_weight(y, device):
    import numpy as np
    import torch
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    weights = np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def cosine_mean(a, b, eps=1e-12):
    import numpy as np
    numerator = (a * b).sum(axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean(numerator / np.maximum(denom, eps)))


@torch.inference_mode()
def predict_representation(model, x, device):
    import numpy as np
    import torch
    model.eval()
    logits_out, z_out = [], []
    for start in range(0, len(x), 512):
        batch = torch.tensor(x[start:start + 512], dtype=torch.float32, device=device)
        logits, z = model(batch, return_z=True)
        logits_out.append(logits.detach().cpu().numpy())
        z_out.append(z.detach().cpu().numpy())
    return np.concatenate(logits_out, axis=0), np.concatenate(z_out, axis=0)

# ===== End compatibility helpers =====


# ===== Metadata / label compatibility helpers =====

def load_feature_metadata(feature_dir):
    import json
    from pathlib import Path

    feature_dir = Path(feature_dir)
    candidates = [
        feature_dir / "feature_metadata.json",
        feature_dir / "metadata.json",
        feature_dir / "features_metadata.json",
    ]

    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

    return {}


def _parse_label_cell(value):
    import json
    import pandas as pd

    if isinstance(value, list):
        return [str(v) for v in value]

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except Exception:
                pass

        for sep in ["|", ",", ";"]:
            if sep in value:
                return [v.strip() for v in value.split(sep) if v.strip()]

        return [value]

    return [str(value)]


def build_label_mapping(metadata_df, feature_meta=None):
    feature_meta = feature_meta or {}

    # Highest priority: use explicit label_id mapping from metadata.
    # Your metadata has columns: label, label_id.
    if "label" in metadata_df.columns and "label_id" in metadata_df.columns:
        pairs = (
            metadata_df[["label", "label_id"]]
            .dropna()
            .drop_duplicates()
            .sort_values("label_id")
        )

        mapping = {}
        for _, row in pairs.iterrows():
            mapping[str(row["label"])] = int(row["label_id"])

        if mapping:
            return mapping

    for key in ["class_names", "classes", "labels", "label_names"]:
        if key in feature_meta and isinstance(feature_meta[key], list):
            return {str(name): idx for idx, name in enumerate(feature_meta[key])}

    label_columns = [
        "genres",
        "genre",
        "labels",
        "label",
        "class_name",
        "category",
        "primary_label",
    ]

    labels = []
    for col in label_columns:
        if col in metadata_df.columns:
            for value in metadata_df[col].tolist():
                labels.extend(_parse_label_cell(value))

    if not labels:
        raise ValueError(
            "Cannot build label mapping: no usable label column found. "
            "Expected label/label_id or one of genres/genre/labels/class_name/category."
        )

    unique = sorted(set(labels))
    return {name: idx for idx, name in enumerate(unique)}


def build_multilabel_targets(metadata_df, split_meta, label_to_id):
    import numpy as np
    import pandas as pd

    n_labels = max(label_to_id.values()) + 1

    class_names = [None] * n_labels
    for name, idx in label_to_id.items():
        class_names[idx] = name

    for i in range(n_labels):
        if class_names[i] is None:
            class_names[i] = f"class_{i}"

    label_columns = [
        "genres",
        "genre",
        "labels",
        "label",
        "class_name",
        "category",
        "primary_label",
    ]

    labels_by_split = {}

    for split, meta in split_meta.items():
        y = np.zeros((len(meta), n_labels), dtype=np.float32)

        # Highest priority: explicit label_id column.
        if "label_id" in meta.columns:
            for i, value in enumerate(meta["label_id"].tolist()):
                if pd.isna(value):
                    continue

                label_id = int(value)
                if 0 <= label_id < n_labels:
                    y[i, label_id] = 1.0

            labels_by_split[split] = y
            continue

        label_col = None
        for col in label_columns:
            if col in meta.columns:
                label_col = col
                break

        if label_col is None:
            raise ValueError(
                f"Cannot build targets for split={split}: no label_id or label column found."
            )

        for i, value in enumerate(meta[label_col].tolist()):
            labels = _parse_label_cell(value)
            for label in labels:
                label = str(label)
                if label in label_to_id:
                    y[i, label_to_id[label]] = 1.0

        labels_by_split[split] = y

    return labels_by_split, class_names

# ===== End metadata / label compatibility helpers =====
