from __future__ import annotations

from .model import Atom, Variable


def query_frontiers(
    answer_vars: tuple[Variable, ...],
    body: tuple[Atom, ...],
) -> tuple[tuple[str, ...], ...]:
    """Compute the query-frontier variables of every source CQ atom.

    Frontier(A_i,Q) contains exactly the variables of A_i that are either
    original query answers or occur in at least one other source CQ atom.
    The returned variable order follows each source atom.
    """
    answer_names = {v.name for v in answer_vars}
    atom_vars: list[tuple[str, ...]] = []
    occurrence_count: dict[str, int] = {}

    for atom in body:
        names = []
        seen = set()
        for term in atom.terms:
            if isinstance(term, Variable) and term.name not in seen:
                seen.add(term.name)
                names.append(term.name)
        atom_vars.append(tuple(names))
        for name in seen:
            occurrence_count[name] = occurrence_count.get(name, 0) + 1

    return tuple(
        tuple(
            name
            for name in names
            if name in answer_names or occurrence_count.get(name, 0) > 1
        )
        for names in atom_vars
    )
