"""Compare query structure before collecting new training labels."""
from collections import Counter, defaultdict

from .core import Const, is_var


def _body(query):
    return tuple(dict.fromkeys(query.body))


def _signature(query):
    def term(value):
        if is_var(value):
            return ('variable',)
        return ('symbol', value.name) if isinstance(value, Const) else ('integer', value)

    atoms = Counter((atom.pred, tuple(term(t) for t in atom.args)) for atom in _body(query))
    head_names = {}
    head = tuple(head_names.setdefault(name, len(head_names)) for name in query.head)
    return head, frozenset(atoms.items())


def equivalent_queries(left, right):
    """Exact alpha-equivalence, preserving answer order and typed constants.

    Body atoms may be reordered. This is variable renaming, not general CQ
    minimization or equivalence under an ontology. Matching avoids enumerating
    every permutation of the whole query body.
    """
    if _signature(left) != _signature(right):
        return False
    forward, inverse = {}, {}
    for source, target in zip(left.head, right.head):
        if source in forward and forward[source] != target:
            return False
        if target in inverse and inverse[target] != source:
            return False
        forward[source], inverse[target] = target, source
    lhs, rhs = _body(left), _body(right)

    def match_atom(source, target, mapping, reverse):
        if source.pred != target.pred or len(source.args) != len(target.args):
            return None
        mapping, reverse = dict(mapping), dict(reverse)
        for a, b in zip(source.args, target.args):
            if is_var(a):
                if not is_var(b) or (a in mapping and mapping[a] != b) or (b in reverse and reverse[b] != a):
                    return None
                mapping[a], reverse[b] = b, a
            elif is_var(b) or type(a) is not type(b) or a != b:
                return None
        return mapping, reverse

    def search(remaining, available, mapping, reverse):
        if not remaining:
            return True
        index = max(remaining, key=lambda i: sum(not is_var(t) or t in mapping for t in lhs[i].args))
        for other in available:
            matched = match_atom(lhs[index], rhs[other], mapping, reverse)
            if matched is not None and search(tuple(i for i in remaining if i != index),
                    tuple(i for i in available if i != other), *matched):
                return True
        return False

    return search(tuple(range(len(lhs))), tuple(range(len(rhs))), forward, inverse)


def filter_training_queries(queries, holdout_queries):
    """Return nonoverlapping unique candidates and explicit exclusion records."""
    holdout, retained = defaultdict(list), defaultdict(list)
    for qid, query in holdout_queries:
        holdout[_signature(query)].append((qid, query))
    result, excluded = [], []
    for qid, query in queries:
        signature = _signature(query)
        duplicate = next((other_id for other_id, other in holdout[signature]
                          if equivalent_queries(query, other)), None)
        reason = 'holdout_equivalent'
        if duplicate is None:
            duplicate = next((other_id for other_id, other in retained[signature]
                              if equivalent_queries(query, other)), None)
            reason = 'training_duplicate'
        if duplicate is not None:
            excluded.append({'qid': qid, 'reason': reason, 'equivalent_to': duplicate})
        else:
            result.append((qid, query))
            retained[signature].append((qid, query))
    return result, excluded
