"""Shared Datalog terms, facts and rule-composition operations.

String terms are variables. Integers and Const values are constants.
Facts stores predicate-indexed sets of ground tuples.
"""
import hashlib
from collections import defaultdict
from dataclasses import dataclass

class Atom:
    __slots__ = ('pred', 'args')

    def __init__(self, pred, args):
        self.pred = pred
        self.args = tuple(args)

    def __eq__(self, other):
        return isinstance(other, Atom) and self.pred == other.pred and (self.args == other.args)

    def __hash__(self):
        return hash((self.pred, self.args))

    def __repr__(self):
        return '%s(%s)' % (self.pred, ','.join(repr(t) if isinstance(t, Const) else str(t) for t in self.args))

class Rule:
    __slots__ = ('head', 'body')

    def __init__(self, head, body):
        self.head = head
        self.body = tuple(body)

    def __eq__(self, other):
        return isinstance(other, Rule) and self.head == other.head and (self.body == other.body)

    def __hash__(self):
        return hash((self.head, self.body))

    def __repr__(self):
        body = ', '.join((repr(a) for a in self.body)) if self.body else 'true'
        return '%s :- %s' % (self.head, body)

@dataclass(frozen=True)
class Const:
    """A symbolic constant, distinct from every numeric constant.

    SQL IDs belong to a database-local ConstantCodec, never to the term itself.
    int(symbol) retains the old fingerprint only for legacy callers; that
    fingerprint is not an identity or a collision-free database ID.
    """
    name: str

    def __post_init__(self):
        object.__setattr__(self, 'name', str(self.name))

    def __int__(self):
        h = hashlib.sha1(('cot-const:' + self.name).encode('utf-8')).hexdigest()
        return -(int(h[:15], 16) + 1)

    def __str__(self):
        return self.name

    def __repr__(self):
        return repr(self.name)

    def __reduce__(self):
        return (Const, (self.name,))


def term_sort_key(term):
    """Deterministic ordering without conflating term kinds or display text."""
    if isinstance(term, Const):
        return ('symbol', term.name)
    if isinstance(term, str):
        return ('variable', term)
    if isinstance(term, int) and not isinstance(term, bool):
        return ('integer', term)
    raise TypeError('expected a variable, integer, or Const')


def ground_row_key(row):
    return tuple(term_sort_key(term) for term in row)


class ConstantCodec:
    """Stable, reversible int64 IDs for one database context.

    Ordinary int64 node IDs pass through. Symbols, large integers, and numbers
    that collide with an allocated symbolic ID receive unused negative codes.
    Encoding new facts or query constants never changes previously issued IDs.
    No dictionary entry is needed for the usual nonnegative integer IDs.
    """
    def __init__(self):
        self._encoded = {}
        self._decoded = {}
        self._negative_used = set()
        self._next = -1

    def _allocate(self, value):
        while self._next in self._negative_used:
            self._next -= 1
        if self._next < -(1 << 63):
            raise OverflowError('constant dictionary exhausted int64 IDs')
        code = self._next
        self._next -= 1
        self._negative_used.add(code)
        self._encoded[value] = code
        self._decoded[code] = value
        return code

    def encode(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, Const)):
            raise TypeError('ground constants must be integers or Const values')
        if isinstance(value, int) and 0 <= value < (1 << 63):
            return value
        if value in self._encoded:
            return self._encoded[value]
        if isinstance(value, int) and -(1 << 63) <= value < 0 and value not in self._decoded:
            self._negative_used.add(value)
            return value
        return self._allocate(value)

    def decode(self, code):
        code = int(code)
        return self._decoded.get(code, code)

def is_var(term):
    return isinstance(term, str)

def atom_vars(atom):
    return {t for t in atom.args if is_var(t)}

def rename_rule(rule, suffix):
    """Rename every variable with a fresh suffix (standardising apart)."""
    mapping = {}

    def tr(term):
        if is_var(term):
            if term not in mapping:
                mapping[term] = '%s@%s' % (term, suffix)
            return mapping[term]
        return term
    head = Atom(rule.head.pred, tuple((tr(t) for t in rule.head.args)))
    body = tuple((Atom(a.pred, tuple((tr(t) for t in a.args))) for a in rule.body))
    return Rule(head, body)
_FRESH = [0]

def fresh_rule(rule):
    _FRESH[0] += 1
    return rename_rule(rule, 'f%d' % _FRESH[0])

def canon_rule(rule):
    """Canonical variable names (V0, V1, ... by first occurrence), body as set."""
    mapping = {}

    def tr(term):
        if is_var(term):
            if term not in mapping:
                mapping[term] = 'V%d' % len(mapping)
            return mapping[term]
        return term
    head = Atom(rule.head.pred, tuple((tr(t) for t in rule.head.args)))
    body = frozenset((Atom(a.pred, tuple((tr(t) for t in a.args))) for a in rule.body))
    ordered = tuple(sorted(body, key=lambda a: (a.pred, ground_row_key(a.args))))
    return Rule(head, ordered)

def apply_atom(atom, subst):
    return Atom(atom.pred, tuple((subst.get(t, t) if is_var(t) else t for t in atom.args)))

def unify_terms(x, y, subst):
    if is_var(x):
        if x in subst:
            return unify_terms(subst[x], y, subst)
        if is_var(y) and y in subst:
            return unify_terms(x, subst[y], subst)
        if x == y:
            return True
        subst[x] = y
        return True
    if is_var(y):
        return unify_terms(y, x, subst)
    return x == y

def unify_atoms(a, b):
    """MGU of two atoms or None (no function symbols, so no occurs check)."""
    if a.pred != b.pred or len(a.args) != len(b.args):
        return None
    subst = {}
    for x, y in zip(a.args, b.args):
        if not unify_terms(x, y, subst):
            return None
    return subst

class Facts:

    def __init__(self, tables=None):
        self.rows = defaultdict(set)
        self._idx = {}
        if tables:
            for pred, rows in tables.items():
                self.rows[pred] = set((tuple(r) for r in rows))

    def add(self, pred, args):
        rows = self.rows[pred]
        before = len(rows)
        rows.add(tuple(args))
        if len(rows) != before:
            self._idx = {k: v for k, v in self._idx.items() if k[0] != pred}
            return True
        return False

    def remove(self, pred, args):
        rows = self.rows[pred]
        if tuple(args) in rows:
            rows.remove(tuple(args))
            self._idx = {k: v for k, v in self._idx.items() if k[0] != pred}
            return True
        return False

    def has(self, pred, args):
        return tuple(args) in self.rows.get(pred, ())

    def facts(self):
        for pred, rows in self.rows.items():
            for r in rows:
                yield (pred, r)

    def size(self):
        return sum((len(r) for r in self.rows.values()))

    def copy(self):
        out = Facts()
        for pred, rows in self.rows.items():
            out.rows[pred] = set(rows)
        return out

    def col_index(self, pred, col):
        key = (pred, col)
        idx = self._idx.get(key)
        if idx is None:
            idx = defaultdict(set)
            for row in self.rows[pred]:
                idx[row[col]].add(row)
            self._idx[key] = idx
        return idx

def _candidates(atom, binding, facts):
    rows = facts.rows.get(atom.pred)
    if not rows:
        return ()
    for pos, term in enumerate(atom.args):
        if is_var(term) and term in binding:
            return facts.col_index(atom.pred, pos).get(binding[term], ())
    return rows

def _join_order(body, facts):
    n = len(body)
    remaining = set(range(n))
    order = []
    bound = set()
    while remaining:

        def score(i):
            atom = body[i]
            n_const = sum((1 for t in atom.args if not is_var(t)))
            shared = len(atom_vars(atom) & bound)
            table = len(facts.rows.get(atom.pred, ()))
            return (n_const, shared, -table)
        i = max(remaining, key=score)
        remaining.discard(i)
        order.append(i)
        bound |= atom_vars(body[i])
    return order

def _pick_next(body, remaining, binding, facts):
    """Greedy adaptive join order: pick the remaining atom with the fewest
    candidate rows under the current binding (index lookups only)."""
    best_pos = None
    best_rows = None
    for pos in remaining:
        rows = _candidates(body[pos], binding, facts)
        n = len(rows)
        if best_rows is None or n < best_rows:
            best_pos, best_rows = (pos, n)
            if n <= 1:
                break
    best_pos = best_pos if best_pos is not None else next(iter(remaining))
    return (best_pos, _candidates(body[best_pos], binding, facts))

def _use_adaptive(body, facts):
    """Use the adaptive order only for bigger joins (>=5 atoms); small joins
    keep the cheap static order to avoid per-node planning overhead."""
    return len(body) >= 5

def enumerate_derivations(rule, facts, limit=None):
    """Yield (head_fact, ground_body_tuple) for every ground instance.

    ``ground_body`` keeps the rule body order (needed by the reasoning-path
    unfolding); duplicate facts collapse naturally as a set at call sites.
    """
    body = rule.body
    adaptive = _use_adaptive(body, facts)
    results = []

    def emit(binding, gbody):
        head_args = tuple((binding.get(t, t) if is_var(t) else t for t in rule.head.args))
        if any((is_var(t) for t in head_args)):
            return
        results.append(((rule.head.pred, head_args), tuple(gbody)))

    def apply_row(atom, row, binding):
        b2 = dict(binding)
        for t, v in zip(atom.args, row):
            if is_var(t):
                if t in b2:
                    if b2[t] != v:
                        return None
                else:
                    b2[t] = v
            elif t != v:
                return None
        return b2

    def step_adaptive(remaining, binding, gbody):
        if limit is not None and len(results) >= limit:
            return
        if not remaining:
            emit(binding, gbody)
            return
        pos, rows = _pick_next(body, remaining, binding, facts)
        atom = body[pos]
        rest = remaining - {pos}
        for row in rows:
            b2 = apply_row(atom, row, binding)
            if b2 is None:
                continue
            gbody[pos] = (atom.pred, row)
            step_adaptive(rest, b2, gbody)
            if limit is not None and len(results) >= limit:
                return

    def step_static(k, binding, gbody):
        if limit is not None and len(results) >= limit:
            return
        if k == len(order):
            emit(binding, gbody)
            return
        pos = order[k]
        atom = body[pos]
        for row in _candidates(atom, binding, facts):
            b2 = apply_row(atom, row, binding)
            if b2 is None:
                continue
            gbody[pos] = (atom.pred, row)
            step_static(k + 1, b2, gbody)
            if limit is not None and len(results) >= limit:
                return
    if adaptive:
        step_adaptive(frozenset(range(len(body))), {}, [None] * len(body))
    else:
        order = _join_order(body, facts)
        step_static(0, {}, [None] * len(body))
    return results

def _match_terms(pattern, target, subst):
    if is_var(pattern):
        if pattern in subst:
            return subst[pattern] == target
        subst[pattern] = target
        return True
    return pattern == target

def _match_atom(pattern, target, subst):
    if pattern.pred != target.pred or len(pattern.args) != len(target.args):
        return False
    for p, t in zip(pattern.args, target.args):
        if not _match_terms(p, t, subst):
            return False
    return True

def subsumes(r1, r2):
    """Paper Definition: exists sigma on r1's variables s.t.
    sigma(Head(r1)) == Head(r2) and sigma(Body(r1)) subset of Body(r2).

    Note: sigma only substitutes variables of r1 (the subsuming rule); it may
    NOT substitute variables of r2.  Bodies are sets, target atoms may be
    reused by cancelled duplicates.
    """
    r1 = rename_rule(r1, 'sub')
    subst = {}
    if not _match_atom(r1.head, r2.head, subst):
        return False

    def rec(i, s):
        if i == len(r1.body):
            return True
        pattern = r1.body[i]
        for target in r2.body:
            s2 = dict(s)
            if _match_atom(pattern, target, s2) and rec(i + 1, s2):
                return True
        return False
    return rec(0, subst)

def remove_redundant(rules):
    """Iteratively prune using only subsumers that are still retained."""
    rules = list(dict.fromkeys((canon_rule(r) for r in rules)))
    # Reverse traversal keeps the first representative of mutually subsuming
    # rules, while a stricter rule may be removed by a later general rule.
    for i in range(len(rules) - 1, -1, -1):
        if any(i != j and subsumes(other, rules[i]) for j, other in enumerate(rules)):
            rules.pop(i)
    return rules

class CQ:
    __slots__ = ('head', 'body')

    def __init__(self, head, body):
        self.head = tuple(head)
        self.body = tuple(body)

    def __eq__(self, other):
        return isinstance(other, CQ) and self.head == other.head and (set(self.body) == set(other.body))

    def __hash__(self):
        return hash((self.head, frozenset(self.body)))

    def __repr__(self):
        return '?- %s' % (', '.join((repr(a) for a in self.body)) or 'true')
