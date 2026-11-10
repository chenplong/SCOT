"""Evaluate the production cost-sensitive policy tree."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path


def load_model(path=None):
    path = path or os.environ.get("SCOT_POLICY_MODEL")
    if path is None:
        path = Path(__file__).resolve().parents[3] / "datasets/s2747.model.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _eval_tree(node, features):
    if not isinstance(node, dict):
        raise ValueError('invalid policy-tree node')
    if node.get('leaf') or 'mode' in node:
        mode = node.get('selected_mode', node.get('mode'))
        if str(mode).lower() not in {'p00', 'p01', 'p10', 'p11'}:
            raise ValueError('policy-tree leaf has no valid selected_mode')
        return str(mode).lower()
    feature = node.get("feature")
    if feature is None:
        raise ValueError('policy-tree split has no feature')
    value = features.get(feature, 0.0)
    if value is None:
        value = 0.0
    threshold = node.get("threshold", 0)
    child = node.get("left") if value <= threshold else node.get("right")
    return _eval_tree(child, features)


def choose_mode(features, model=None):
    if model is None:
        model = load_model()
    if isinstance(model, (str, os.PathLike)):
        model = load_model(model)
    if not isinstance(model, dict):
        raise ValueError('policy model must be a JSON mapping')
    prepared = dict(features)
    medians = model.get('feature_imputation_values', {})
    for name in set(model.get('feature_names', ())) | set(medians):
        value = prepared.get(name)
        try:
            finite = value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        prepared[name] = float(value) if finite else float(medians.get(name, 0.0))
    return _eval_tree(model.get('tree', model), prepared)
