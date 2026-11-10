from __future__ import annotations
from collections import defaultdict
from .model import Atom, Constant, QueryItem, Rule, Variable
from .substitution import apply_atom, apply_term, standardize_rule, unify_atoms


class CandidateGenerator:
    """Generate the original-atom branch plus all applicable COT branches."""

    def __init__(self, cot_rules: tuple[Rule, ...]):
        self.index = defaultdict(list)
        for rule in cot_rules:
            self.index[(rule.head.predicate, rule.head.arity)].append(rule)

    @staticmethod
    def _is_identity(source_rule: Rule) -> bool:
        if len(source_rule.body) != 1:
            return False
        head = source_rule.head
        body = source_rule.body[0]
        return (
            head.predicate == body.predicate
            and head.arity == body.arity
            and head.terms == body.terms
        )

    def for_atom(
        self,
        atom: Atom,
        index: int,
        *,
        exposed_vars: tuple[str, ...] | None = None,
        forbidden_names=(),
    ) -> tuple[QueryItem, ...]:
        """Return alternatives for one source atom.

        ``exposed_vars`` are the query-frontier variables that this atom must
        expose to the rest of the CQ. Variables occurring only inside a COT
        body are existential and never leave the branch.
        """
        if exposed_vars is None:
            seen = set()
            interface = []
            for term in atom.terms:
                if isinstance(term, Variable) and term.name not in seen:
                    seen.add(term.name)
                    interface.append(term.name)
            exposed_vars = tuple(interface)

        out = [QueryItem(index, (atom,), 'ORIGINAL', None, tuple(exposed_vars))]
        for n, source_rule in enumerate(self.index.get((atom.predicate, atom.arity), ())):
            if self._is_identity(source_rule):
                continue
            forbidden = set(forbidden_names) | {
                term.name for term in atom.terms if isinstance(term, Variable)
            }
            rule = standardize_rule(source_rule, f'q{index}_{n}', forbidden)
            substitution = unify_atoms(rule.head, atom)
            if substitution is not None:
                constants = tuple(
                    (v, substitution[v]) for v in exposed_vars
                    if isinstance(substitution.get(v), Constant)
                )
                out.append(
                    QueryItem(
                        index,
                        tuple(apply_atom(a, substitution) for a in rule.body),
                        'COT',
                        source_rule.rule_id,
                        tuple(exposed_vars),
                        constants,
                        tuple((v, apply_term(Variable(v), substitution))
                              for v in exposed_vars),
                    )
                )
        return tuple(out)
