"""Fit the cost-sensitive policy tree used by the query engine."""
from __future__ import annotations
import csv
import glob
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

MODES = ("P00", "P01", "P10", "P11")
COST_COLUMNS = tuple(f"{m}_median_effective_cost_seconds" for m in MODES)

def _expand_inputs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for item in sorted(matches):
            p = Path(item)
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(p)
    if not out:
        raise ValueError("no input CSV files matched --input")
    return out


def _to_float(value: Any) -> float:
    if value is None:
        return math.nan
    s = str(value).strip()
    if not s:
        return math.nan
    if s.lower() in {"true", "yes"}:
        return 1.0
    if s.lower() in {"false", "no"}:
        return 0.0
    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(f"non-numeric feature value: {value!r}") from exc


def _is_valid_row(row: dict[str, str]) -> bool:
    v = str(row.get("valid_for_training", "")).strip().lower()
    return v in {"1", "true", "yes"}


@dataclass
class Dataset:
    X: np.ndarray
    costs: np.ndarray
    feature_names: list[str]
    metadata: list[dict[str, Any]]
    imputation_values: np.ndarray
    missing_feature_counts: dict[str, int]
    unavailable_features: list[str]
    source_files: list[dict[str, str]]


def _load_training_data(paths: list[Path]) -> Dataset:
    rows_all: list[dict[str, str]] = []
    expected_features: list[str] | None = None

    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"{path}: missing CSV header")

            features = sorted(x for x in reader.fieldnames if x.startswith("feat_"))
            if not features:
                raise ValueError(f"{path}: no feat_* columns found")

            missing_costs = [c for c in COST_COLUMNS if c not in reader.fieldnames]
            if missing_costs:
                raise ValueError(f"{path}: missing cost columns: {missing_costs}")

            if expected_features is None:
                expected_features = features
            elif features != expected_features:
                only_here = sorted(set(features) - set(expected_features))
                missing_here = sorted(set(expected_features) - set(features))
                raise ValueError(
                    f"{path}: feature schema mismatch; "
                    f"extra={only_here[:10]} missing={missing_here[:10]}"
                )

            for row in reader:
                if _is_valid_row(row):
                    row = dict(row)
                    row["__source_csv"] = str(path)
                    rows_all.append(row)

    if not rows_all:
        raise ValueError("no valid_for_training rows found")

    assert expected_features is not None
    feature_names = expected_features

    X = np.empty((len(rows_all), len(feature_names)), dtype=np.float64)
    costs = np.empty((len(rows_all), len(MODES)), dtype=np.float64)
    metadata: list[dict[str, Any]] = []

    for i, row in enumerate(rows_all):
        for j, name in enumerate(feature_names):
            X[i, j] = _to_float(row.get(name))

        for k, col in enumerate(COST_COLUMNS):
            value = _to_float(row.get(col))
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"invalid cost at row {i + 1}, {col}: {row.get(col)!r}"
                )
            costs[i, k] = value

        metadata.append(
            {
                "dataset_tag": row.get("dataset_tag"),
                "qid": row.get("qid"),
                "query": row.get("query"),
                "source_csv": row.get("__source_csv"),
                "best_mode_recorded": row.get("best_mode"),
                "label_measurement_protocol": row.get("label_measurement_protocol") or "unspecified",
                "label_cost_schema": row.get("label_cost_schema") or "unspecified",
            }
        )

    missing_counts = {name: int((~np.isfinite(X[:, j])).sum())
                      for j, name in enumerate(feature_names)}
    unavailable = [name for name, count in missing_counts.items() if count == len(rows_all)]
    # Median imputation is fit only on the integrated training set.
    imputation_values = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        finite = X[np.isfinite(X[:, j]), j]
        if finite.size == 0:
            imputation_values[j] = 0.0
        else:
            imputation_values[j] = float(np.median(finite))
        bad = ~np.isfinite(X[:, j])
        if np.any(bad):
            X[bad, j] = imputation_values[j]

    return Dataset(
        X=X,
        costs=costs,
        feature_names=feature_names,
        metadata=metadata,
        imputation_values=imputation_values,
        missing_feature_counts={name: count for name, count in missing_counts.items() if count},
        unavailable_features=unavailable,
        source_files=[{'name': path.name,
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in paths],
    )


def _regret_matrix(costs: np.ndarray, objective: str, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    oracle = np.min(costs, axis=1)

    if objective == "absolute":
        regret = costs - oracle[:, None]
    elif objective == "relative":
        denom = np.maximum(oracle, epsilon)
        regret = costs / denom[:, None] - 1.0
    elif objective == "log":
        regret = np.log((costs + epsilon) / (oracle[:, None] + epsilon))
    else:
        raise ValueError(f"unknown objective: {objective}")

    regret = np.maximum(regret, 0.0)
    return regret, oracle


@dataclass
class Node:
    n_samples: int
    depth: int
    action: int
    loss: float
    action_losses: list[float]
    feature: int | None = None
    threshold: float | None = None
    gain: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class CostSensitivePolicyTree:
    def __init__(
        self,
        *,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        min_gain: float = 1e-9,
        max_thresholds_per_feature: int = 0,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_gain = float(min_gain)
        self.max_thresholds_per_feature = int(max_thresholds_per_feature)
        self.root_: Node | None = None
        self.feature_gain_: np.ndarray | None = None

    @staticmethod
    def _node_action_loss(regret: np.ndarray, indices: np.ndarray,
                          weights: np.ndarray) -> tuple[int, float, np.ndarray]:
        totals = (regret[indices] * weights[indices, None]).sum(axis=0)
        action = int(np.argmin(totals))
        return action, float(totals[action]), totals

    def fit(self, X: np.ndarray, regret: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "CostSensitivePolicyTree":
        self.feature_gain_ = np.zeros(X.shape[1], dtype=np.float64)
        idx = np.arange(X.shape[0], dtype=np.int64)
        weights = (np.ones(X.shape[0], dtype=np.float64)
                   if sample_weight is None
                   else np.asarray(sample_weight, dtype=np.float64))
        if weights.shape[0] != X.shape[0]:
            raise ValueError("sample_weight length mismatch")
        self.root_ = self._build(X, regret, idx, weights, depth=0)
        return self

    def _candidate_positions(self, values: np.ndarray) -> np.ndarray:
        n = values.size
        pos = np.arange(self.min_samples_leaf - 1, n - self.min_samples_leaf)
        if pos.size == 0:
            return pos

        pos = pos[values[pos] < values[pos + 1]]
        if pos.size == 0:
            return pos

        cap = self.max_thresholds_per_feature
        if cap and pos.size > cap:
            pick = np.linspace(0, pos.size - 1, num=cap, dtype=int)
            pos = pos[np.unique(pick)]
        return pos

    def _build(
        self,
        X: np.ndarray,
        regret: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        depth: int,
    ) -> Node:
        action, parent_loss, action_losses = self._node_action_loss(
            regret, indices, weights)
        node = Node(
            n_samples=int(indices.size),
            depth=depth,
            action=action,
            loss=parent_loss,
            action_losses=[float(x) for x in action_losses],
        )

        if depth >= self.max_depth:
            return node
        if indices.size < 2 * self.min_samples_leaf:
            return node
        if parent_loss <= self.min_gain:
            return node

        best_gain = self.min_gain
        best_feature: int | None = None
        best_threshold: float | None = None
        best_left: np.ndarray | None = None
        best_right: np.ndarray | None = None

        node_regret = regret[indices] * weights[indices, None]

        for j in range(X.shape[1]):
            vals = X[indices, j]
            order = np.argsort(vals, kind="mergesort")
            sorted_vals = vals[order]
            positions = self._candidate_positions(sorted_vals)
            if positions.size == 0:
                continue

            sorted_indices = indices[order]
            sorted_regret = node_regret[order]
            csum = np.cumsum(sorted_regret, axis=0)
            total = csum[-1]

            left_totals = csum[positions]
            right_totals = total[None, :] - left_totals
            left_losses = np.min(left_totals, axis=1)
            right_losses = np.min(right_totals, axis=1)

            gains = parent_loss - left_losses - right_losses
            local = int(np.argmax(gains))
            gain = float(gains[local])

            if gain > best_gain:
                p = int(positions[local])
                a = float(sorted_vals[p])
                b = float(sorted_vals[p + 1])
                threshold = a + (b - a) / 2.0
                best_gain = gain
                best_feature = j
                best_threshold = threshold
                best_left = sorted_indices[: p + 1]
                best_right = sorted_indices[p + 1 :]

        if best_feature is None or best_left is None or best_right is None:
            return node

        node.feature = best_feature
        node.threshold = float(best_threshold)
        node.gain = float(best_gain)
        assert self.feature_gain_ is not None
        self.feature_gain_[best_feature] += best_gain

        node.left = self._build(X, regret, best_left, weights, depth + 1)
        node.right = self._build(X, regret, best_right, weights, depth + 1)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("tree is not fitted")
        out = np.empty(X.shape[0], dtype=np.int64)
        for i in range(X.shape[0]):
            node = self.root_
            while not node.is_leaf:
                assert node.feature is not None
                assert node.threshold is not None
                assert node.left is not None and node.right is not None
                if X[i, node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            out[i] = node.action
        return out


def _node_to_dict(node: Node, feature_names: list[str]) -> dict[str, Any]:
    base = {
        "n_samples": node.n_samples,
        "depth": node.depth,
        "selected_mode": MODES[node.action],
        "node_loss": node.loss,
        "action_losses": {MODES[i]: node.action_losses[i] for i in range(len(MODES))},
    }
    if node.is_leaf:
        base["leaf"] = True
        return base

    assert node.feature is not None
    assert node.threshold is not None
    assert node.left is not None and node.right is not None
    base.update(
        {
            "leaf": False,
            "feature": feature_names[node.feature],
            "threshold": node.threshold,
            "gain": node.gain,
            "left": _node_to_dict(node.left, feature_names),
            "right": _node_to_dict(node.right, feature_names),
        }
    )
    return base


def _make_sample_weights(oracle_action, metadata, mode: str, alpha: float):
    n = len(oracle_action)
    if mode == "none" or n == 0 or alpha == 0:
        return np.ones(n, dtype=np.float64)
    keys = []
    for i, action in enumerate(oracle_action):
        if mode == "dataset_class":
            keys.append((str(metadata[i].get("dataset_tag") or ""), int(action)))
        else:
            keys.append(int(action))
    counts = Counter(keys)
    k = max(1, len(counts))
    weights = np.asarray([
        (n / (k * counts[key])) ** float(alpha) for key in keys
    ], dtype=np.float64)
    mean = float(weights.mean())
    if mean > 0:
        weights /= mean
    return weights


def train_model(inputs, output, *, max_depth=20, min_samples_leaf=2,
                min_gain=1e-9, max_thresholds_per_feature=0,
                objective="relative", epsilon=1e-6,
                weight_mode="none", weight_alpha=0.5):
    data = _load_training_data(_expand_inputs([str(p) for p in inputs]))
    regret, _ = _regret_matrix(data.costs, objective, epsilon)
    weights = _make_sample_weights(np.argmin(data.costs, axis=1), data.metadata,
                                   weight_mode, weight_alpha)
    tree = CostSensitivePolicyTree(max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, min_gain=min_gain,
        max_thresholds_per_feature=max_thresholds_per_feature)
    tree.fit(data.X, regret, sample_weight=weights)
    model = {
        "schema_version": 1,
        "model_type": "cost_sensitive_policy_tree",
        "modes": list(MODES),
        "objective": objective,
        "objective_description": {
            "absolute": "sum(selected_cost - empirical_oracle_cost)",
            "relative": "sum(selected_cost / empirical_oracle_cost - 1)",
            "log": "sum(log((selected_cost+epsilon)/(oracle_cost+epsilon)))",
        }[objective],
        "hyperparameters": {"max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf, "min_gain": min_gain,
            "max_thresholds_per_feature": max_thresholds_per_feature,
            "epsilon": epsilon},
        "feature_names": data.feature_names,
        "weighting": {"mode": weight_mode, "alpha": weight_alpha},
        "feature_imputation_values": {name: float(value) for name, value in
            zip(data.feature_names, data.imputation_values)},
        "tree": _node_to_dict(tree.root_, data.feature_names),
        "training_data": {
            "fitting_rows": len(data.X),
            "feature_schema_columns": len(data.feature_names),
            "observed_feature_columns": len(data.feature_names) - len(data.unavailable_features),
            "unavailable_features": data.unavailable_features,
            "missing_feature_counts": data.missing_feature_counts,
            "missing_feature_policy": "Training-column medians; wholly missing columns imputed to zero and cannot split",
            "source_files": data.source_files,
            "measurements_recollected_by_training": False,
            "label_measurement_protocol_counts": dict(Counter(
                row['label_measurement_protocol'] for row in data.metadata)),
            "label_cost_schema_counts": dict(Counter(
                row['label_cost_schema'] for row in data.metadata)),
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"rows": len(data.X), "features": len(data.feature_names), "model": str(output)}
