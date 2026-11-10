"""Pre-execution features for the cost-sensitive mode selector."""
from __future__ import annotations

MODES = ("P00", "P01", "P10", "P11")

PLAN_STAT_KEYS = (
    "atom_groups",
    "candidate_branches",
    "active_candidate_branches",
    "static_empty_branches",
    "frontier_variables_total",
    "candidate_stages",
    "candidate_inner_joins",
    "candidate_semi_joins",
    "candidate_cross_joins",
    "candidate_eliminated_vars",
    "candidate_early_projections",
    "candidate_absorbed_unary_filters",
    "unary_frontier_source_groups",
    "unary_frontier_views",
    "unary_frontier_consolidation_joins",
    "absorbed_unary_views",
    "unary_absorption_edges",
    "residual_unary_views",
    "residual_unary_guards",
    "zero_frontier_groups",
    "top_level_groups_after_absorption",
    "global_inner_joins",
    "global_semi_joins",
    "global_cross_joins",
)


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def query_raw_features(query, frontiers, candidate_groups):
    occurrences, constants, predicates = {}, 0, set()
    for atom in query.body:
        predicates.add(str(atom.predicate))
        for term in atom.terms:
            name = getattr(term, "name", None)
            if name is not None:
                occurrences[str(name)] = occurrences.get(str(name), 0) + 1
            else:
                constants += 1
    branch_counts = [len(group) for group in candidate_groups]
    frontier_sizes = [len(frontier) for frontier in frontiers]
    return {
        "query_atom_count": len(query.body),
        "query_answer_var_count": len(query.answer_vars),
        "query_distinct_var_count": len(occurrences),
        "query_var_occurrence_count": sum(occurrences.values()),
        "query_shared_var_count": sum(1 for n in occurrences.values() if n > 1),
        "query_constant_occurrence_count": constants,
        "query_distinct_predicate_count": len(predicates),
        "query_frontier_size_sum": sum(frontier_sizes),
        "query_frontier_size_max": max(frontier_sizes, default=0),
        "query_zero_frontier_atom_count": sum(1 for n in frontier_sizes if n == 0),
        "query_unary_frontier_atom_count": sum(1 for n in frontier_sizes if n == 1),
        "query_multi_frontier_atom_count": sum(1 for n in frontier_sizes if n > 1),
        "candidate_group_count": len(candidate_groups),
        "candidate_branch_count_sum": sum(branch_counts),
        "candidate_branch_count_max": max(branch_counts, default=0),
        "candidate_branch_count_min": min(branch_counts, default=0),
        "candidate_branch_count_mean": _ratio(sum(branch_counts), len(branch_counts)),
        "cot_candidate_branch_count": sum(max(0, n - 1) for n in branch_counts),
    }


def flatten_features(raw, plans):
    features = dict(raw)
    for mode in MODES:
        plan = plans.get(mode) or {}
        features[f"{mode}_compile_empty"] = int(bool(plan.get("__compile_empty__", False)))
        features[f"{mode}_sql_chars"] = int(plan.get("__sql_chars__", 0))
        for key in PLAN_STAT_KEYS:
            value = plan.get(key, 0)
            features[f"{mode}_{key}"] = float(value) if isinstance(value, float) else int(value or 0)
    p10 = plans.get("P10") or {}
    p01 = plans.get("P01") or {}
    q_branches = int(p10.get("candidate_branches", 0) or 0)
    q_stages = int(p10.get("candidate_stages", 0) or 0)
    q_elim = int(p10.get("candidate_eliminated_vars", 0) or 0)
    q_proj = int(p10.get("candidate_early_projections", 0) or 0)
    atom_groups = int(p01.get("atom_groups", 0) or 0)
    top_groups = int(p01.get("top_level_groups_after_absorption", 0) or 0)
    removed_groups = max(0, atom_groups - top_groups)
    absorbed_filters = int(p01.get("candidate_absorbed_unary_filters", 0) or 0)
    features.update({
        "qf_stage_ratio": _ratio(q_stages, q_branches),
        "qf_eliminated_per_branch": _ratio(q_elim, q_branches),
        "qf_early_projection_per_branch": _ratio(q_proj, q_branches),
        "ufa_removed_top_level_groups": removed_groups,
        "ufa_replication_per_removed_group": _ratio(absorbed_filters, removed_groups),
    })
    return {"feat_" + key: value for key, value in features.items()}
